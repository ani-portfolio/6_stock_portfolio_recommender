from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import sys
from typing import Optional
import pandas as pd

# Add your src path
sys.path.append('..')

# Try to import modules with fallbacks
try:
    from src.parameters import *
    from src.rag import *
    from src.query_router import *
    from src.data_ingestion import *
    from src.recommender import *
    from src.charts import *
    from src.utils import *
    print("✅ Successfully imported all modules")
except ImportError as e:
    print(f"⚠️ Warning: Could not import modules: {e}")
    print("📝 Using fallback functions for development")
    
    # Fallback functions for development
    def router(query):
        return {'route': 'semantic_search'}
    
    def rag_query_stocks(**kwargs):
        return {'success': True, 'answer': 'Mock response for development'}
    
    def recommend_stocks_from_query(**kwargs):
        return {'success': True, 'answer': 'Mock recommendation response'}
    
    def load_table_from_bigquery(*args):
        # Return mock data for development
        return pd.DataFrame({
            'Ticker': ['AAPL', 'MSFT', 'GOOGL'],
            'Company_Name': ['Apple Inc.', 'Microsoft Corp.', 'Alphabet Inc.'],
            'Market_Cap': [3000000, 2800000, 1800000],
            'Risk': ['Low', 'Low', 'Medium'],
            'Sentiment': ['Positive', 'Positive', 'Neutral']
        })
    
    def create_sample_queries():
        pass
    
    def create_summary_chart(df):
        # Return top 3 for mock data
        return df.head(3) if df is not None else None

app = FastAPI(title="Stock Recommendation System", version="1.0.0")

# Create directories if they don't exist
os.makedirs("templates", exist_ok=True)
os.makedirs("media", exist_ok=True)

# Mount static files for media
app.mount("/media", StaticFiles(directory="media"), name="media")

templates = Jinja2Templates(directory="templates")

# Configuration
SAMPLE_QUERIES = [
    "Which tech stocks have high returns and low risk?",
    "Show me dividend stocks with strong fundamentals", 
    "What are the best performing healthcare stocks?",
    "Find stocks with low volatility and steady growth",
    "Which energy stocks are undervalued right now?",
    "Show me emerging market opportunities"
]

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main stock recommendation page"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "sample_queries": SAMPLE_QUERIES
    })

@app.get("/risk-level", response_class=HTMLResponse)
async def risk_level(request: Request):
    """Risk Level page"""
    return templates.TemplateResponse("risk_level.html", {"request": request})

@app.get("/architecture", response_class=HTMLResponse)
async def architecture(request: Request):
    """System Architecture page"""
    return templates.TemplateResponse("architecture.html", {"request": request})

@app.get("/documentation", response_class=HTMLResponse)
async def documentation(request: Request):
    """Documentation page"""
    return templates.TemplateResponse("documentation.html", {"request": request})

@app.get("/links", response_class=HTMLResponse)
async def links(request: Request):
    """Links page"""
    return templates.TemplateResponse("links.html", {"request": request})

@app.get("/api/charts-data")
async def get_charts_data():
    """API endpoint to get charts data"""
    try:
        # Get configuration with defaults
        dataset_id = globals().get('dataset_id', 'your_dataset')
        table_id = globals().get('table_id', 'your_table')
        project_id = globals().get('project_id', 'your_project')
        
        # Load stock data
        df_stock_data = load_table_from_bigquery(dataset_id, table_id, project_id)
        
        if df_stock_data is not None and not df_stock_data.empty:
            # Create summary chart
            df_summary = create_summary_chart(df_stock_data)
            
            if df_summary is not None and not df_summary.empty:
                # Clean column names (replace underscores with spaces)
                df_clean = df_summary.copy()
                df_clean.columns = df_clean.columns.str.replace('_', ' ')
                
                # Convert to JSON-friendly format
                return JSONResponse(content={
                    "success": True,
                    "data": df_clean.to_dict('records'),
                    "columns": list(df_clean.columns)
                })
            else:
                return JSONResponse(content={
                    "success": False, 
                    "error": "No summary data could be generated"
                })
        else:
            return JSONResponse(content={
                "success": False,
                "error": "No stock data available from BigQuery"
            })
            
    except Exception as e:
        return JSONResponse(content={
            "success": False,
            "error": f"Data loading failed: {str(e)}"
        })

@app.post("/analyze")
async def analyze_stock_query(user_query: str = Form(...)):
    """Analyze stock query and return recommendations"""
    
    if not user_query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Route the query
        route_result = router(user_query)
        route = route_result.get('route', 'unknown')
        
        if route == 'semantic_search':
            # Use semantic search with RAG
            response = rag_query_stocks(
                query=user_query,
                top_k=10,
                groq_llm_model=globals().get('groq_llm_model', 'mixtral-8x7b-32768'),
                huggingface_embeddings_model=globals().get('huggingface_embeddings_model', 'all-MiniLM-L6-v2'),
                pinecone_index_name=globals().get('pinecone_index_name', 'stock-index')
            )
            
        elif route == 'recommender':
            # Use ML recommender
            df_stock_data = load_table_from_bigquery(
                globals().get('dataset_id', 'your_dataset'),
                globals().get('table_id', 'your_table'), 
                globals().get('project_id', 'your_project')
            )
            
            if df_stock_data is not None:
                # Define column types
                non_numerical_columns = [
                    'Ticker', 'Company_Name', 'Sector', 'Industry',
                    'Country', 'Business_Summary', 'Sentiment', 'Update_Date'
                ]
                numerical_columns = [col for col in df_stock_data.columns 
                                   if col not in non_numerical_columns]
                
                response = recommend_stocks_from_query(
                    df=df_stock_data,
                    user_query=user_query,
                    numerical_columns=numerical_columns,
                    non_numerical_columns=non_numerical_columns,
                    top_n=5,
                    groq_api_key=globals().get('get_api_key', lambda x: 'your_api_key')("GROQ_API_KEY")
                )
            else:
                response = {"success": False, "error": "Could not load stock data"}
        else:
            response = {"success": False, "error": f"Unknown route: {route}"}
        
        # Add metadata to response
        response["query"] = user_query
        response["route"] = route
        
        return JSONResponse(content=response)
        
    except Exception as e:
        return JSONResponse(content={
            "query": user_query,
            "success": False,
            "error": f"Analysis failed: {str(e)}",
            "route": "error"
        })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Stock Recommendation System"}

@app.get("/api/check-architecture-image")
async def check_architecture_image():
    """Check if architecture image exists"""
    image_path = "media/architecture.png"
    if os.path.exists(image_path):
        return JSONResponse(content={"exists": True, "path": "/media/architecture.png"})
    else:
        return JSONResponse(content={"exists": False, "message": "Place your architecture.png in the media/ folder"})

@app.get("/api/check-media-files")
async def check_media_files():
    """Check what files exist in media folder"""
    media_files = []
    if os.path.exists("media"):
        for file in os.listdir("media"):
            file_path = os.path.join("media", file)
            if os.path.isfile(file_path):
                media_files.append(file)
    
    return JSONResponse(content={
        "media_folder_exists": os.path.exists("media"),
        "files": media_files,
        "total_files": len(media_files)
    })

@app.get("/api/check-link-images")
async def check_link_images():
    """Check if link images exist"""
    images = {
        "github": None,
        "portfolio": None, 
        "linkedin": None
    }
    
    # Check for various file extensions
    extensions = ['.png', '.jpg', '.jpeg', '.gif', '.svg', '']
    
    for image_name in images.keys():
        for ext in extensions:
            image_path = f"media/{image_name}{ext}"
            if os.path.exists(image_path):
                images[image_name] = f"/media/{image_name}{ext}"
                break
    
    return JSONResponse(content=images)
    
    # Generate ticker rows
    def create_ticker_row(queries, direction='left'):
        # Quadruple the items to ensure no gaps during animation
        items = ''.join([f'<div class="ticker-item" onclick="setQuery(\'{q}\')">{q}</div>' for q in queries * 4])
        return f'<div class="ticker-container"><div class="ticker-content ticker-{direction}">{items}</div></div>'
    
    ticker_rows = f"""
    <div class="space-y-4 mb-8">
        {create_ticker_row(ticker_queries['row1'], 'left')}
        {create_ticker_row(ticker_queries['row2'], 'right')}
        {create_ticker_row(ticker_queries['row3'], 'left')}
    </div>
    """
    
if __name__ == "__main__":
    import uvicorn
    
    # Create media directory and check for architecture image
    if not os.path.exists("media/architecture.png"):
        print("⚠️ Architecture image not found")
        print("📁 Create a 'media' folder and place your architecture.png file there")
        print("🖼️ The image will be automatically displayed once added")
    else:
        print("✅ Architecture image found")
    
    # Try to create sample queries
    try:
        create_sample_queries()
        print("✅ Sample queries created")
    except Exception as e:
        print(f"⚠️ Could not create sample queries: {e}")
    
    print("🚀 Starting Stock Recommendation System...")
    print("📍 Home page: http://localhost:8000/")
    print("⚠️ Risk Level page: http://localhost:8000/risk-level")
    print("📚 Documentation page: http://localhost:8000/documentation")
    print("🏗️ Architecture page: http://localhost:8000/architecture")
    print("🔗 Links page: http://localhost:8000/links")
    print("🔧 Health check: http://localhost:8000/health")
    
    # Run the application
    uvicorn.run(
        "__main__:app",
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        reload_dirs=[".", "../src"],
        reload_includes=["*.py", "*.html"],
        log_level="info"
    )