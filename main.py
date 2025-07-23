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

def create_template_files():
    """Create HTML template files with embedded CSS and JavaScript"""
    
    # Shared styles
    shared_styles = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
        
        /* Navigation styles */
        .nav-item {
            position: relative;
            padding: 0.75rem 1.5rem;
            color: #000;
            text-decoration: none;
            font-weight: 500;
            transition: color 0.3s ease;
            cursor: pointer;
        }
        
        .nav-item::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 0;
            height: 2px;
            background-color: rgba(107, 33, 168, 0.8);
            box-shadow: 0 4px 20px 0 rgba(107, 33, 168, 0.6);
            transition: width 0.3s ease;
        }
        
        .nav-item:hover::after,
        .nav-item.active::after {
            width: 100%;
        }
        
        .nav-item.active {
            color: #6b21a8;
        }
        
        /* Ticker animation styles */
        .ticker-container {
            overflow: hidden;
            white-space: nowrap;
            position: relative;
        }
        
        .ticker-content {
            display: inline-block;
            animation-timing-function: linear;
            animation-iteration-count: infinite;
        }
        
        .ticker-item {
            display: inline-block;
            margin-right: 1rem;
            cursor: pointer;
            transition: all 0.2s ease;
            white-space: nowrap;
            font-size: 1rem;
            color: black;
            position: relative;
        }
        
        .ticker-item::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 0;
            height: 2px;
            background-color: rgba(107, 33, 168, 0.8);
            box-shadow: 0 4px 20px 0 rgba(107, 33, 168, 0.6);
            transition: width 0.3s ease;
        }
        
        .ticker-item:hover::after {
            width: 100%;
        }
        
        .ticker-left {
            animation: tickerLeft 90s linear infinite;
        }
        
        .ticker-right {
            animation: tickerRight 90s linear infinite;
        }
        
        @keyframes tickerLeft {
            0% { transform: translateX(0); }
            100% { transform: translateX(-100%); }
        }
        
        @keyframes tickerRight {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(0); }
        }
        
        .ticker-container:hover .ticker-content {
            animation-play-state: paused;
        }
        
        /* Table styles */
        .data-table {
            border-collapse: collapse;
            width: 100%;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .data-table th,
        .data-table td {
            padding: 12px;
            text-align: center;
            border-bottom: 1px solid #e5e7eb;
        }
        
        .data-table th {
            background-color: #f8f9fa;
            font-weight: 600;
            color: #374151;
        }
        
        .data-table tbody tr:hover {
            background-color: #f9fafb;
        }
        
        /* Risk and sentiment styling */
        .risk-high { background-color: #fee2e2; color: #dc2626; border-radius: 4px; padding: 2px 6px; }
        .risk-medium { background-color: #fef3c7; color: #d97706; border-radius: 4px; padding: 2px 6px; }
        .risk-low { background-color: #dcfce7; color: #16a34a; border-radius: 4px; padding: 2px 6px; }
        
        .sentiment-positive { background-color: #dcfce7; color: #16a34a; border-radius: 4px; padding: 2px 6px; }
        .sentiment-negative { background-color: #fee2e2; color: #dc2626; border-radius: 4px; padding: 2px 6px; }
        .sentiment-neutral { background-color: #f3f4f6; color: #6b7280; border-radius: 4px; padding: 2px 6px; }
        
        /* Architecture diagram styles */
        .architecture-container {
            display: flex;
            justify-content: center;
            margin: 20px 0;
            border: 2px solid #ddd;
            border-radius: 10px;
            padding: 20px;
            background: linear-gradient(135deg, #f9f9f9 0%, #ffffff 100%);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        .architecture-image {
            max-width: 100%;
            height: auto;
            image-rendering: -webkit-optimize-contrast;
            image-rendering: -moz-crisp-edges;
            image-rendering: crisp-edges;
            border-radius: 8px;
            box-shadow: 0 6px 16px rgba(0,0,0,0.15);
            transition: transform 0.3s ease;
            cursor: pointer;
        }
        
        .architecture-image:hover {
            transform: scale(1.02);
        }
        
        /* Fullscreen modal styles */
        .fullscreen-modal {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.9);
            z-index: 1000;
            display: none;
            align-items: center;
            justify-content: center;
            cursor: pointer;
        }
        
        .fullscreen-modal.active {
            display: flex;
        }
        
        .fullscreen-image {
            max-width: 95%;
            max-height: 95%;
            object-fit: contain;
            border-radius: 8px;
            box-shadow: 0 0 50px rgba(255, 255, 255, 0.1);
        }
        
        .close-button {
            position: absolute;
            top: 20px;
            right: 30px;
            color: white;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
            z-index: 1001;
            transition: opacity 0.3s ease;
        }
        
        .close-button:hover {
            opacity: 0.7;
        }
        
        .architecture-placeholder {
            width: 100%;
            height: 400px;
            background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
            border-radius: 8px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: #6b7280;
            border: 2px dashed #d1d5db;
        }
        
        .architecture-components {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        
        .component-card {
            background: white;
            border-radius: 12px;
            padding: 24px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        
        .component-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 16px rgba(0,0,0,0.15);
        }
        
        .component-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .component-description {
            color: #6b7280;
            line-height: 1.6;
        }
        
        .tech-badge {
            display: inline-block;
            background: #f3f4f6;
            color: #374151;
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 500;
            margin: 2px;
        }
        
        /* Documentation styles */
        .doc-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 24px;
            margin: 32px 0;
        }
        
        .doc-card {
            background: rgba(255, 255, 255, 0.6);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
            height: fit-content;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        }
        
        .doc-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.15);
            border-color: #d1d5db;
        }
        
        .doc-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .doc-purpose {
            font-weight: 500;
            color: #374151;
            margin-bottom: 16px;
        }
        
        .doc-links {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        
        .doc-links li {
            margin: 8px 0;
        }
        
        .doc-links a {
            color: #3b82f6;
            text-decoration: none;
            font-size: 0.9rem;
            transition: color 0.2s ease;
        }
        
        .doc-links a:hover {
            color: #1d4ed8;
            text-decoration: underline;
        }
        
        .doc-links a::before {
            content: '→ ';
            margin-right: 4px;
        }
        
        /* Links page styles */
        .links-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 32px;
            margin: 48px 0;
            max-width: 1200px;
            margin-left: auto;
            margin-right: auto;
        }
        
        .link-card {
            background: white;
            border-radius: 20px;
            padding: 32px;
            border: 2px solid #e5e7eb;
            box-shadow: 0 8px 24px rgba(0,0,0,0.08);
            transition: all 0.4s ease;
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        
        .link-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #3b82f6, #8b5cf6, #06b6d4);
            transform: translateX(-100%);
            transition: transform 0.4s ease;
        }
        
        .link-card:hover {
            transform: translateY(-8px) scale(1.02);
            box-shadow: 0 16px 40px rgba(0,0,0,0.15);
            border-color: #3b82f6;
        }
        
        .link-card:hover::before {
            transform: translateX(0);
        }
        
        .link-icon {
            font-size: 3rem;
            margin-bottom: 16px;
            display: block;
        }
        
        .link-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: #1f2937;
            margin-bottom: 12px;
        }
        
        .link-url {
            display: inline-block;
            background: linear-gradient(135deg, #3b82f6, #1d4ed8);
            color: white;
            padding: 12px 24px;
            border-radius: 50px;
            text-decoration: none;
            font-weight: 600;
            margin-bottom: 16px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        }
        
        .link-url:hover {
            background: linear-gradient(135deg, #1d4ed8, #1e40af);
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(59, 130, 246, 0.4);
            color: white;
            text-decoration: none;
        }
        
        .link-description {
            color: #6b7280;
            line-height: 1.6;
            font-size: 0.95rem;
        }
        
        .creator-footer {
            margin-top: 64px;
            padding: 24px;
            background: white;
            border-radius: 16px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }
        
        /* Link tile hover effects */
        .link-tile-image {
            filter: blur(3px);
            transition: all 0.4s ease;
        }
        
        .link-card:hover .link-tile-image {
            filter: blur(0);
        }
    </style>
    """
    
    # Navigation bar component
    nav_bar = """
    <div class="flex justify-center py-4" style="background: #f0f0f5;">
        <nav class="flex">
            <a href="/" class="nav-item" id="nav-home">HOME</a>
            <a href="/risk-level" class="nav-item" id="nav-risk-level">RISK LEVEL</a>
            <div class="nav-item">CHARTS</div>
            <a href="/documentation" class="nav-item" id="nav-documentation">DOCUMENTATION</a>
            <a href="/architecture" class="nav-item" id="nav-architecture">SYSTEM ARCHITECTURE</a>
            <a href="/links" class="nav-item" id="nav-links">LINKS</a>
        </nav>
    </div>
    """
    
    # Ticker queries data
    ticker_queries = {
        'row1': [
            'Which tech stocks have high returns and low risk?',
            'Show me dividend stocks with strong fundamentals',
            'What are the best performing healthcare stocks?',
            'Find stocks with low volatility and steady growth',
            'Which energy stocks are undervalued right now?',
            'Show me emerging market opportunities'
        ],
        'row2': [
            'Analyze Tesla stock performance this quarter',
            'Compare Apple vs Microsoft fundamentals', 
            'Best ESG stocks for sustainable investing',
            'Cryptocurrency stocks vs traditional finance',
            'Small cap stocks with growth potential',
            'REITs with high dividend yields'
        ],
        'row3': [
            'Portfolio diversification strategies for 2025',
            'Impact of interest rates on stock market',
            'Blue chip stocks for conservative investors',
            'Biotech stocks with FDA approvals pending',
            'Electric vehicle sector analysis',
            'AI and machine learning stock picks'
        ]
    }
    
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
    
    # Home page template
    index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Recommendation System</title>
    <script src="https://cdn.tailwindcss.com"></script>
    {shared_styles}
</head>
<body class="font-sans min-h-screen" style="background: #f0f0f5;">
    <div class="flex flex-col h-screen">
        {nav_bar}

        <div class="flex-1 flex flex-col">
            <main class="flex-1 flex flex-col">
                <!-- Welcome Section -->
                <div id="welcomeSection" class="flex-1 flex items-center justify-center px-6">
                    <div class="max-w-4xl w-full mx-auto">
                        <div class="text-center mb-12">
                            <h2 class="text-4xl md:text-5xl font-semibold text-gray-800 mb-6 leading-tight">
                                Get answers. Find insights.<br>
                                Make smarter investments.
                            </h2>
                        </div>
                        
                        <div class="relative mb-8">
                            <form id="analysisForm" class="relative">
                                <textarea 
                                    id="userQuery"
                                    name="user_query"
                                    placeholder="Ask about stocks, sectors, or portfolio strategies..."
                                    rows="3"
                                    required
                                    class="w-full px-4 py-3 pr-12 rounded-2xl border border-white/30 text-black placeholder-gray-600 focus:outline-none resize-none shadow-lg"
                                    style="background: rgba(255, 255, 255, 0.3); backdrop-filter: blur(15px); -webkit-backdrop-filter: blur(15px); box-shadow: 0 8px 40px 0 rgba(31, 38, 135, 0.2);"
                                ></textarea>
                                <button 
                                    type="submit"
                                    id="analyzeBtn"
                                    class="absolute right-2 bottom-2 px-3 py-1 bg-black hover:bg-gray-800 text-white text-sm rounded-lg transition-colors"
                                >
                                    <span id="btnIcon">Ask</span>
                                    <div id="btnSpinner" class="hidden">
                                        <svg class="animate-spin h-4 w-4" viewBox="0 0 24 24">
                                            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"></circle>
                                            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                        </svg>
                                    </div>
                                </button>
                            </form>
                        </div>

                        {ticker_rows}
                    </div>
                </div>

                <!-- Results Section -->
                <div id="resultsSection" class="hidden flex-1 px-6 py-8">
                    <div class="max-w-4xl mx-auto w-full">
                        <div class="bg-white rounded-lg border border-gray-200 p-6">
                            <div class="flex items-center justify-between mb-6">
                                <h3 class="text-lg font-medium text-gray-800">Analysis Results</h3>
                                <span id="routeIndicator" class="px-2 py-1 rounded text-xs font-medium bg-gray-100 text-gray-600"></span>
                            </div>
                            
                            <div id="resultsContent" class="prose prose-sm max-w-none"></div>
                            
                            <div class="mt-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                                <div class="flex items-center space-x-2">
                                    <svg class="w-4 h-4 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z"></path>
                                    </svg>
                                    <p class="text-yellow-800 text-sm">This information is not financial advice and should not be considered as such</p>
                                </div>
                            </div>
                            
                            <button onclick="newQuery()" class="mt-6 px-4 py-2 bg-black hover:bg-gray-800 text-white rounded-md text-sm transition-colors">New Query</button>
                        </div>
                    </div>
                </div>

                <!-- Error Section -->
                <div id="errorSection" class="hidden flex-1 px-6 py-8">
                    <div class="max-w-4xl mx-auto w-full">
                        <div class="bg-white rounded-lg border border-red-200 p-6">
                            <div class="flex items-center space-x-3 mb-4">
                                <svg class="w-5 h-5 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                                </svg>
                                <h3 class="text-lg font-medium text-red-800">Analysis Failed</h3>
                            </div>
                            <div id="errorContent" class="text-red-700"></div>
                            <p class="text-gray-600 text-sm mt-4">Please check your API keys and ensure all dependencies are properly configured.</p>
                            <button onclick="newQuery()" class="mt-6 px-4 py-2 bg-black hover:bg-gray-800 text-white rounded-md text-sm transition-colors">Try Again</button>
                        </div>
                    </div>
                </div>
            </main>
        </div>
    </div>

    <script>
        // Set active navigation
        document.getElementById('nav-home').classList.add('active');

        function setQuery(query) {{
            document.getElementById('userQuery').value = query;
        }}

        function newQuery() {{
            document.getElementById('userQuery').value = '';
            document.getElementById('welcomeSection').classList.remove('hidden');
            document.getElementById('resultsSection').classList.add('hidden');
            document.getElementById('errorSection').classList.add('hidden');
        }}

        function showLoading() {{
            document.getElementById('btnIcon').classList.add('hidden');
            document.getElementById('btnSpinner').classList.remove('hidden');
            document.getElementById('analyzeBtn').disabled = true;
        }}

        function hideLoading() {{
            document.getElementById('btnIcon').classList.remove('hidden');
            document.getElementById('btnSpinner').classList.add('hidden');
            document.getElementById('analyzeBtn').disabled = false;
        }}

        function formatMarkdown(text) {{
            return text
                .replace(/\\n/g, '<br>')
                .replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>')
                .replace(/\\*(.*?)\\*/g, '<em>$1</em>')
                .replace(/`(.*?)`/g, '<code class="bg-gray-100 px-1 rounded text-sm">$1</code>');
        }}

        function showResults(data) {{
            const resultsContent = document.getElementById('resultsContent');
            const routeIndicator = document.getElementById('routeIndicator');
            
            routeIndicator.textContent = data.route === 'semantic_search' ? 'Semantic Search' : 'AI Recommender';
            
            resultsContent.innerHTML = `
                <div class="space-y-6">
                    <div class="bg-gray-50 p-4 rounded-lg">
                        <h4 class="font-medium text-gray-800 mb-2">Query:</h4>
                        <p class="text-gray-600">${{data.query}}</p>
                    </div>
                    <div class="bg-gray-50 p-4 rounded-lg">
                        <h4 class="font-medium text-gray-800 mb-2">Analysis:</h4>
                        <div class="text-gray-800 leading-relaxed">${{formatMarkdown(data.answer)}}</div>
                    </div>
                </div>
            `;
            
            document.getElementById('welcomeSection').classList.add('hidden');
            document.getElementById('resultsSection').classList.remove('hidden');
            document.getElementById('errorSection').classList.add('hidden');
        }}

        function showError(error) {{
            document.getElementById('errorContent').textContent = error;
            document.getElementById('welcomeSection').classList.add('hidden');
            document.getElementById('errorSection').classList.remove('hidden');
            document.getElementById('resultsSection').classList.add('hidden');
        }}

        document.getElementById('analysisForm').addEventListener('submit', async function(e) {{
            e.preventDefault();
            
            const userQuery = document.getElementById('userQuery').value.trim();
            if (!userQuery) return;
            
            showLoading();
            
            try {{
                const formData = new FormData();
                formData.append('user_query', userQuery);
                
                const response = await fetch('/analyze', {{
                    method: 'POST',
                    body: formData
                }});
                
                const data = await response.json();
                
                if (data.success) {{
                    showResults(data);
                }} else {{
                    showError(data.error || 'Analysis failed');
                }}
                
            }} catch (error) {{
                showError(`Network error: ${{error.message}}`);
            }} finally {{
                hideLoading();
            }}
        }});
    </script>
</body>
</html>"""
    
    # Risk Level page template
    risk_level_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Recommendation System</title>
    <script src="https://cdn.tailwindcss.com"></script>
    {shared_styles}
</head>
<body class="font-sans min-h-screen" style="background: #f0f0f5;">
    <div class="flex flex-col h-screen">
        {nav_bar}

        <div class="flex-1 flex flex-col">
            <main class="flex-1 px-6 py-8">
                <div class="max-w-7xl mx-auto w-full">
                    <!-- Loading State -->
                    <div id="loadingState" class="flex items-center justify-center py-12">
                        <div class="flex items-center space-x-3">
                            <svg class="animate-spin h-6 w-6 text-gray-600" viewBox="0 0 24 24">
                                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"></circle>
                                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            <span class="text-gray-600">Loading stock data...</span>
                        </div>
                    </div>

                    <!-- Charts Content -->
                    <div id="chartsContent" class="hidden">
                        <div class="bg-white rounded-xl border border-gray-100 shadow-xl overflow-hidden">
                            <div class="overflow-x-auto">
                                <table id="stockTable" class="data-table">
                                    <thead id="tableHeader"></thead>
                                    <tbody id="tableBody"></tbody>
                                </table>
                            </div>
                        </div>
                    </div>

                    <!-- Error State -->
                    <div id="errorState" class="hidden">
                        <div class="bg-white rounded-lg border border-red-200 p-6 text-center">
                            <div class="flex items-center justify-center space-x-3 mb-4">
                                <svg class="w-6 h-6 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                                </svg>
                                <h3 class="text-lg font-medium text-red-800">❌ No data available to display</h3>
                            </div>
                            <p id="errorMessage" class="text-red-700 mb-4"></p>
                            <p class="text-gray-600 text-sm">Please check your data source and try again.</p>
                            <button onclick="loadChartsData()" class="mt-4 px-4 py-2 bg-black hover:bg-gray-800 text-white rounded-md text-sm transition-colors">Retry</button>
                        </div>
                    </div>
                </div>
            </main>
        </div>
    </div>

    <script>
        // Set active navigation
        document.getElementById('nav-risk-level').classList.add('active');

        function getRiskClass(value) {{
            if (typeof value === 'string') {{
                const lower = value.toLowerCase();
                if (lower.includes('high')) return 'risk-high';
                if (lower.includes('medium') || lower.includes('moderate')) return 'risk-medium';
                if (lower.includes('low')) return 'risk-low';
            }}
            return '';
        }}

        function getSentimentClass(value) {{
            if (typeof value === 'string') {{
                const lower = value.toLowerCase();
                if (lower.includes('positive') || lower.includes('bullish')) return 'sentiment-positive';
                if (lower.includes('negative') || lower.includes('bearish')) return 'sentiment-negative';
                if (lower.includes('neutral')) return 'sentiment-neutral';
            }}
            return '';
        }}

        function formatNumber(value) {{
            if (typeof value === 'number') {{
                if (value > 1000000) {{
                    return (value / 1000000).toFixed(1) + 'M';
                }} else if (value > 1000) {{
                    return (value / 1000).toFixed(1) + 'K';
                }} else {{
                    return value.toLocaleString(undefined, {{maximumFractionDigits: 1}});
                }}
            }}
            return value;
        }}

        function createTable(data, columns) {{
            const tableHeader = document.getElementById('tableHeader');
            const tableBody = document.getElementById('tableBody');
            
            // Clear existing content
            tableHeader.innerHTML = '';
            tableBody.innerHTML = '';
            
            // Create header with improved styling
            const headerRow = document.createElement('tr');
            columns.forEach(column => {{
                const th = document.createElement('th');
                th.textContent = column;
                th.style.cssText = `
                    padding: 8px 12px;
                    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                    font-weight: 600;
                    color: #475569;
                    font-size: 0.75rem;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                    border-bottom: 2px solid #e2e8f0;
                    text-align: center;
                `;
                headerRow.appendChild(th);
            }});
            tableHeader.appendChild(headerRow);
            
            // Create body rows with enhanced styling
            data.forEach((row, index) => {{
                const tr = document.createElement('tr');
                tr.style.cssText = `
                    transition: all 0.2s ease;
                    font-size: 0.875rem;
                `;
                
                tr.addEventListener('mouseenter', function() {{
                    this.style.background = 'linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%)';
                    this.style.transform = 'scale(1.002)';
                    this.style.boxShadow = '0 2px 8px rgba(0,0,0,0.1)';
                }});
                
                tr.addEventListener('mouseleave', function() {{
                    this.style.background = '';
                    this.style.transform = '';
                    this.style.boxShadow = '';
                }});
                
                columns.forEach((column, colIndex) => {{
                    const td = document.createElement('td');
                    const value = row[column];
                    
                    // Base cell styling
                    td.style.cssText = `
                        padding: 8px 12px;
                        text-align: center;
                        border-bottom: 1px solid #f1f5f9;
                        vertical-align: middle;
                    `;
                    
                    // Apply special styling based on column type and position
                    if (colIndex === 0) {{
                        // Ticker column
                        td.style.cssText += `
                            font-weight: 700;
                            color: #1e293b;
                            font-size: 0.9rem;
                        `;
                        td.textContent = value;
                    }} else if (colIndex === 1) {{
                        // Company name column
                        td.style.cssText += `
                            font-weight: 500;
                            color: #475569;
                            text-align: left;
                            font-size: 0.8rem;
                        `;
                        td.textContent = value;
                    }} else if (typeof value === 'number') {{
                        // Number columns
                        td.style.cssText += `
                            font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
                            font-weight: 500;
                        `;
                        td.textContent = formatNumber(value);
                    }} else {{
                        // Other columns
                        td.textContent = formatNumber(value);
                    }}
                    
                    // Apply badge styling for risk and sentiment
                    if (column.toLowerCase().includes('risk')) {{
                        const riskClass = getRiskClass(value);
                        if (riskClass) {{
                            let badgeStyle = `
                                border-radius: 6px;
                                padding: 3px 8px;
                                font-weight: 600;
                                font-size: 0.7rem;
                                text-transform: uppercase;
                                letter-spacing: 0.5px;
                                display: inline-block;
                            `;
                            
                            if (riskClass === 'risk-high') {{
                                badgeStyle += `
                                    background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
                                    color: #dc2626;
                                    border: 1px solid #fca5a5;
                                `;
                            }} else if (riskClass === 'risk-medium') {{
                                badgeStyle += `
                                    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
                                    color: #d97706;
                                    border: 1px solid #fbbf24;
                                `;
                            }} else if (riskClass === 'risk-low') {{
                                badgeStyle += `
                                    background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
                                    color: #16a34a;
                                    border: 1px solid #86efac;
                                `;
                            }}
                            
                            td.innerHTML = `<span style="${{badgeStyle}}">${{value}}</span>`;
                        }}
                    }} else if (column.toLowerCase().includes('sentiment')) {{
                        const sentimentClass = getSentimentClass(value);
                        if (sentimentClass) {{
                            let badgeStyle = `
                                border-radius: 6px;
                                padding: 3px 8px;
                                font-weight: 600;
                                font-size: 0.7rem;
                                text-transform: uppercase;
                                letter-spacing: 0.5px;
                                display: inline-block;
                            `;
                            
                            if (sentimentClass === 'sentiment-positive') {{
                                badgeStyle += `
                                    background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
                                    color: #16a34a;
                                    border: 1px solid #86efac;
                                `;
                            }} else if (sentimentClass === 'sentiment-negative') {{
                                badgeStyle += `
                                    background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
                                    color: #dc2626;
                                    border: 1px solid #fca5a5;
                                `;
                            }} else if (sentimentClass === 'sentiment-neutral') {{
                                badgeStyle += `
                                    background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
                                    color: #6b7280;
                                    border: 1px solid #d1d5db;
                                `;
                            }}
                            
                            td.innerHTML = `<span style="${{badgeStyle}}">${{value}}</span>`;
                        }}
                    }}
                    
                    tr.appendChild(td);
                }});
                tableBody.appendChild(tr);
            }});
        }}

        async function loadChartsData() {{
            // Show loading state
            document.getElementById('loadingState').classList.remove('hidden');
            document.getElementById('chartsContent').classList.add('hidden');
            document.getElementById('errorState').classList.add('hidden');
            
            try {{
                const response = await fetch('/api/charts-data');
                const result = await response.json();
                
                if (result.success && result.data && result.data.length > 0) {{
                    // Show charts content
                    createTable(result.data, result.columns);
                    document.getElementById('loadingState').classList.add('hidden');
                    document.getElementById('chartsContent').classList.remove('hidden');
                }} else {{
                    // Show error state
                    document.getElementById('errorMessage').textContent = result.error || 'No data available';
                    document.getElementById('loadingState').classList.add('hidden');
                    document.getElementById('errorState').classList.remove('hidden');
                }}
            }} catch (error) {{
                // Show error state
                document.getElementById('errorMessage').textContent = `Network error: ${{error.message}}`;
                document.getElementById('loadingState').classList.add('hidden');
                document.getElementById('errorState').classList.remove('hidden');
            }}
        }}

        // Load data when page loads
        document.addEventListener('DOMContentLoaded', loadChartsData);
    </script>
</body>
</html>"""
    charts_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Recommendation System - Charts</title>
    <script src="https://cdn.tailwindcss.com"></script>
    {shared_styles}
</head>
<body class="font-sans min-h-screen" style="background: #f0f0f5;">
    <div class="flex flex-col h-screen">
        {nav_bar}

        <div class="flex-1 flex flex-col">
            <main class="flex-1 px-6 py-8">
                <div class="max-w-7xl mx-auto w-full">
                    <!-- Page Header -->
                    <div class="text-center mb-8">
                        <h1 class="text-4xl font-semibold text-black mb-2">RISK LEVEL</h1>
                        <p class="text-gray-600">Top 25 Stocks by Market Cap</p>
                    </div>

                    <!-- Loading State -->
                    <div id="loadingState" class="flex items-center justify-center py-12">
                        <div class="flex items-center space-x-3">
                            <svg class="animate-spin h-6 w-6 text-gray-600" viewBox="0 0 24 24">
                                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"></circle>
                                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            <span class="text-gray-600">Loading stock data...</span>
                        </div>
                    </div>

                    <!-- Charts Content -->
                    <div id="chartsContent" class="hidden">
                        <div class="bg-white rounded-lg border border-gray-200 shadow-lg overflow-hidden">
                            <div class="overflow-x-auto">
                                <table id="stockTable" class="data-table">
                                    <thead id="tableHeader"></thead>
                                    <tbody id="tableBody"></tbody>
                                </table>
                            </div>
                        </div>
                        
                        <div class="text-center mt-6 text-gray-500">
                            <p>Top 25 Stocks by Market Cap</p>
                        </div>
                    </div>

                    <!-- Error State -->
                    <div id="errorState" class="hidden">
                        <div class="bg-white rounded-lg border border-red-200 p-6 text-center">
                            <div class="flex items-center justify-center space-x-3 mb-4">
                                <svg class="w-6 h-6 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                                </svg>
                                <h3 class="text-lg font-medium text-red-800">❌ No data available to display</h3>
                            </div>
                            <p id="errorMessage" class="text-red-700 mb-4"></p>
                            <p class="text-gray-600 text-sm">Please check your data source and try again.</p>
                            <button onclick="loadChartsData()" class="mt-4 px-4 py-2 bg-black hover:bg-gray-800 text-white rounded-md text-sm transition-colors">Retry</button>
                        </div>
                    </div>
                </div>
            </main>
        </div>
    </div>

    <script>
        // Set active navigation
        document.getElementById('nav-charts').classList.add('active');

        function getRiskClass(value) {{
            if (typeof value === 'string') {{
                const lower = value.toLowerCase();
                if (lower.includes('high')) return 'risk-high';
                if (lower.includes('medium') || lower.includes('moderate')) return 'risk-medium';
                if (lower.includes('low')) return 'risk-low';
            }}
            return '';
        }}

        function getSentimentClass(value) {{
            if (typeof value === 'string') {{
                const lower = value.toLowerCase();
                if (lower.includes('positive') || lower.includes('bullish')) return 'sentiment-positive';
                if (lower.includes('negative') || lower.includes('bearish')) return 'sentiment-negative';
                if (lower.includes('neutral')) return 'sentiment-neutral';
            }}
            return '';
        }}

        function formatNumber(value) {{
            if (typeof value === 'number') {{
                return value.toLocaleString(undefined, {{maximumFractionDigits: 1}});
            }}
            return value;
        }}

        function createTable(data, columns) {{
            const tableHeader = document.getElementById('tableHeader');
            const tableBody = document.getElementById('tableBody');
            
            // Clear existing content
            tableHeader.innerHTML = '';
            tableBody.innerHTML = '';
            
            // Create header
            const headerRow = document.createElement('tr');
            columns.forEach(column => {{
                const th = document.createElement('th');
                th.textContent = column;
                headerRow.appendChild(th);
            }});
            tableHeader.appendChild(headerRow);
            
            // Create body rows
            data.forEach(row => {{
                const tr = document.createElement('tr');
                columns.forEach(column => {{
                    const td = document.createElement('td');
                    const value = row[column];
                    
                    // Format the value
                    td.textContent = formatNumber(value);
                    
                    // Apply styling based on column type
                    if (column.toLowerCase().includes('risk')) {{
                        const riskClass = getRiskClass(value);
                        if (riskClass) td.classList.add(riskClass);
                    }} else if (column.toLowerCase().includes('sentiment')) {{
                        const sentimentClass = getSentimentClass(value);
                        if (sentimentClass) td.classList.add(sentimentClass);
                    }}
                    
                    tr.appendChild(td);
                }});
                tableBody.appendChild(tr);
            }});
        }}

        async function loadChartsData() {{
            // Show loading state
            document.getElementById('loadingState').classList.remove('hidden');
            document.getElementById('chartsContent').classList.add('hidden');
            document.getElementById('errorState').classList.add('hidden');
            
            try {{
                const response = await fetch('/api/charts-data');
                const result = await response.json();
                
                if (result.success && result.data && result.data.length > 0) {{
                    // Show charts content
                    createTable(result.data, result.columns);
                    document.getElementById('loadingState').classList.add('hidden');
                    document.getElementById('chartsContent').classList.remove('hidden');
                }} else {{
                    // Show error state
                    document.getElementById('errorMessage').textContent = result.error || 'No data available';
                    document.getElementById('loadingState').classList.add('hidden');
                    document.getElementById('errorState').classList.remove('hidden');
                }}
            }} catch (error) {{
                // Show error state
                document.getElementById('errorMessage').textContent = `Network error: ${{error.message}}`;
                document.getElementById('loadingState').classList.add('hidden');
                document.getElementById('errorState').classList.remove('hidden');
            }}
        }}

        // Load data when page loads
        document.addEventListener('DOMContentLoaded', loadChartsData);
    </script>
</body>
</html>"""
    
    
    # Architecture page template
    architecture_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Recommendation System</title>
    <script src="https://cdn.tailwindcss.com"></script>
    {shared_styles}
</head>
<body class="font-sans min-h-screen" style="background: #f0f0f5;">
    <div class="flex flex-col h-screen">
        {nav_bar}

        <div class="flex-1 flex flex-col">
            <main class="flex-1 px-6 py-8">
                <div class="max-w-7xl mx-auto w-full">
                    <!-- Architecture Diagram -->
                    <div class="architecture-container">
                        <div id="architectureDiagram" class="w-full">
                            <!-- Image will be loaded here or placeholder shown -->
                            <div class="architecture-placeholder">
                                <svg class="w-16 h-16 mb-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path>
                                </svg>
                                <h3 class="text-lg font-medium mb-2">System Architecture Diagram</h3>
                                <p class="text-sm">Architecture diagram will be displayed here</p>
                                <p class="text-xs mt-2">Place your architecture.png in the media/ folder</p>
                            </div>
                        </div>
                    </div>

                    <!-- Fullscreen Modal -->
                    <div id="fullscreenModal" class="fullscreen-modal">
                        <span class="close-button" onclick="closeFullscreen()">&times;</span>
                        <img id="fullscreenImage" class="fullscreen-image" alt="Architecture Diagram Fullscreen">
                    </div>
                </div>
            </main>
        </div>
    </div>

    <script>
        // Set active navigation
        document.getElementById('nav-architecture').classList.add('active');

        // Try to load architecture image
        async function loadArchitectureImage() {{
            try {{
                const response = await fetch('/api/check-architecture-image');
                const result = await response.json();
                
                if (result.exists) {{
                    const diagramContainer = document.getElementById('architectureDiagram');
                    diagramContainer.innerHTML = `
                        <img src="${{result.path}}" 
                             class="architecture-image"
                             alt="System Architecture Diagram"
                             onclick="openFullscreen('${{result.path}}')"
                             onload="console.log('Architecture image loaded successfully')"
                             onerror="showImageError()">
                    `;
                }} else {{
                    showImageNotFound(result.message);
                }}
            }} catch (error) {{
                console.error('Error checking architecture image:', error);
                showImageError();
            }}
        }}

        function showImageNotFound(message) {{
            const diagramContainer = document.getElementById('architectureDiagram');
            diagramContainer.innerHTML = `
                <div class="architecture-placeholder">
                    <svg class="w-16 h-16 mb-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
                    </svg>
                    <h3 class="text-lg font-medium mb-2 text-orange-600">Architecture Diagram</h3>
                    <p class="text-sm text-gray-600 mb-2">${{message || 'Image not found'}}</p>
                    <p class="text-xs text-gray-500">Create a 'media' folder and place your architecture.png file there</p>
                    <button onclick="loadArchitectureImage()" class="mt-3 px-3 py-1 bg-gray-600 hover:bg-gray-700 text-white text-xs rounded transition-colors">
                        Retry
                    </button>
                </div>
            `;
        }}

        function showImageError() {{
            showImageNotFound('Error loading architecture image');
        }}

        function openFullscreen(imageSrc) {{
            const modal = document.getElementById('fullscreenModal');
            const fullscreenImage = document.getElementById('fullscreenImage');
            fullscreenImage.src = imageSrc;
            modal.classList.add('active');
            document.body.style.overflow = 'hidden';
        }}

        function closeFullscreen() {{
            const modal = document.getElementById('fullscreenModal');
            modal.classList.remove('active');
            document.body.style.overflow = 'auto';
        }}

        // Close fullscreen when clicking on the modal background
        document.getElementById('fullscreenModal').addEventListener('click', function(e) {{
            if (e.target === this) {{
                closeFullscreen();
            }}
        }});

        // Close fullscreen with Escape key
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'Escape') {{
                closeFullscreen();
            }}
        }});

        // Try to load image when page loads
        document.addEventListener('DOMContentLoaded', loadArchitectureImage);
    </script>
</body>
</html>"""
    
    
    # Documentation page template
    documentation_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Recommendation System</title>
    <script src="https://cdn.tailwindcss.com"></script>
    {shared_styles}
</head>
<body class="font-sans min-h-screen" style="background: #f0f0f5;">
    <div class="flex flex-col h-screen">
        {nav_bar}

        <div class="flex-1 flex flex-col">
            <main class="flex-1 px-6 py-8 overflow-y-auto">
                <div class="max-w-7xl mx-auto w-full">
                    <!-- Documentation Grid -->
                    <div class="doc-grid">
                        <!-- Row 1 -->
                        <div class="doc-card">
                            <h4 class="doc-title">
                                HuggingFace Embeddings
                            </h4>
                            <p class="doc-purpose">Text-to-vector conversion for semantic search</p>
                            <ul class="doc-links">
                                <li><a href="https://huggingface.co/" target="_blank">HuggingFace Hub</a></li>
                                <li><a href="https://huggingface.co/sentence-transformers" target="_blank">Sentence Transformers</a></li>
                                <li><a href="https://huggingface.co/blog/getting-started-with-embeddings" target="_blank">Embeddings Guide</a></li>
                            </ul>
                        </div>

                        <div class="doc-card">
                            <h4 class="doc-title">
                                Pinecone Vector Database
                            </h4>
                            <p class="doc-purpose">High-performance vector similarity search</p>
                            <ul class="doc-links">
                                <li><a href="https://docs.pinecone.io/" target="_blank">Pinecone Documentation</a></li>
                                <li><a href="https://docs.pinecone.io/guides/getting-started/quickstart" target="_blank">Getting Started Guide</a></li>
                                <li><a href="https://docs.pinecone.io/guides/data/understanding-multitenancy" target="_blank">Python SDK</a></li>
                            </ul>
                        </div>

                        <div class="doc-card">
                            <h4 class="doc-title">
                                LangChain ChatGroq
                            </h4>
                            <p class="doc-purpose">Fast language model inference</p>
                            <ul class="doc-links">
                                <li><a href="https://python.langchain.com/docs/get_started/introduction" target="_blank">LangChain Documentation</a></li>
                                <li><a href="https://python.langchain.com/docs/integrations/chat/groq" target="_blank">ChatGroq Integration</a></li>
                                <li><a href="https://console.groq.com/docs/quickstart" target="_blank">Groq API Docs</a></li>
                            </ul>
                        </div>

                        <!-- Row 2 -->
                        <div class="doc-card">
                            <h4 class="doc-title">
                                GitHub Actions
                            </h4>
                            <p class="doc-purpose">CI/CD automation platform</p>
                            <ul class="doc-links">
                                <li><a href="https://docs.github.com/en/actions" target="_blank">GitHub Actions Docs</a></li>
                                <li><a href="https://docs.github.com/en/actions/using-workflows" target="_blank">Workflow Syntax</a></li>
                                <li><a href="https://github.com/google-github-actions" target="_blank">GCP Integration</a></li>
                            </ul>
                        </div>

                        <div class="doc-card">
                            <h4 class="doc-title">
                                Cloud Run
                            </h4>
                            <p class="doc-purpose">Serverless container platform</p>
                            <ul class="doc-links">
                                <li><a href="https://cloud.google.com/run/docs" target="_blank">Cloud Run Documentation</a></li>
                                <li><a href="https://cloud.google.com/run/docs/quickstarts" target="_blank">Quickstart Guide</a></li>
                                <li><a href="https://cloud.google.com/run/docs/deploying" target="_blank">Deployment Guide</a></li>
                            </ul>
                        </div>

                        <div class="doc-card">
                            <h4 class="doc-title">
                                Artifact Registry
                            </h4>
                            <p class="doc-purpose">Container and package management</p>
                            <ul class="doc-links">
                                <li><a href="https://cloud.google.com/artifact-registry/docs" target="_blank">Artifact Registry Docs</a></li>
                                <li><a href="https://cloud.google.com/artifact-registry/docs/docker" target="_blank">Docker Guide</a></li>
                                <li><a href="https://cloud.google.com/artifact-registry/docs/analysis" target="_blank">Security Features</a></li>
                            </ul>
                        </div>

                        <!-- Row 3 -->
                        <div class="doc-card">
                            <h4 class="doc-title">
                                BigQuery
                            </h4>
                            <p class="doc-purpose">Data warehouse and analytics platform</p>
                            <ul class="doc-links">
                                <li><a href="https://cloud.google.com/bigquery/docs" target="_blank">BigQuery Documentation</a></li>
                                <li><a href="https://cloud.google.com/bigquery/docs/quickstarts" target="_blank">Getting Started</a></li>
                                <li><a href="https://cloud.google.com/bigquery/docs/reference/standard-sql" target="_blank">SQL Reference</a></li>
                            </ul>
                        </div>

                        <div class="doc-card">
                            <h4 class="doc-title">
                                Docker
                            </h4>
                            <p class="doc-purpose">Containerization platform</p>
                            <ul class="doc-links">
                                <li><a href="https://docs.docker.com/" target="_blank">Docker Documentation</a></li>
                                <li><a href="https://docs.docker.com/engine/reference/builder/" target="_blank">Dockerfile Reference</a></li>
                                <li><a href="https://docs.docker.com/develop/dev-best-practices/" target="_blank">Best Practices</a></li>
                            </ul>
                        </div>

                        <div class="doc-card">
                            <h4 class="doc-title">
                                Prefect
                            </h4>
                            <p class="doc-purpose">Modern workflow orchestration</p>
                            <ul class="doc-links">
                                <li><a href="https://docs.prefect.io/" target="_blank">Prefect Documentation</a></li>
                                <li><a href="https://docs.prefect.io/latest/getting-started/" target="_blank">Getting Started</a></li>
                                <li><a href="https://docs.prefect.io/latest/cloud/" target="_blank">Cloud Platform</a></li>
                            </ul>
                        </div>

                        <!-- Row 4 -->
                        <div class="doc-card">
                            <h4 class="doc-title">
                                Streamlit
                            </h4>
                            <p class="doc-purpose">Web application framework for data science</p>
                            <ul class="doc-links">
                                <li><a href="https://docs.streamlit.io/" target="_blank">Streamlit Documentation</a></li>
                                <li><a href="https://docs.streamlit.io/library/api-reference" target="_blank">API Reference</a></li>
                                <li><a href="https://streamlit.io/gallery" target="_blank">Gallery & Examples</a></li>
                                <li><a href="https://docs.streamlit.io/streamlit-community-cloud" target="_blank">Deployment Guide</a></li>
                            </ul>
                        </div>

                        <div class="doc-card">
                            <h4 class="doc-title">
                                Yahoo Finance
                            </h4>
                            <p class="doc-purpose">Financial data provider</p>
                            <ul class="doc-links">
                                <li><a href="https://pypi.org/project/yfinance/" target="_blank">yfinance Library</a></li>
                                <li><a href="https://finance.yahoo.com/" target="_blank">Yahoo Finance</a></li>
                                <li><a href="https://github.com/ranaroussi/yfinance" target="_blank">GitHub Repository</a></li>
                                <li><a href="https://github.com/ranaroussi/yfinance#quick-start" target="_blank">Usage Examples</a></li>
                            </ul>
                        </div>
                    </div>
                    <div class="text-center mt-12 text-gray-500">
                        <p>📚 Complete documentation for the RAG-LLM Stock Portfolio Recommender System</p>
                    </div>
                </div>
            </main>
        </div>
    </div>

    <script>
        // Set active navigation
        document.getElementById('nav-documentation').classList.add('active');
    </script>
</body>
</html>"""

    
    # Links page template
    links_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Recommendation System</title>
    <script src="https://cdn.tailwindcss.com"></script>
    {shared_styles}
</head>
<body class="font-sans min-h-screen" style="background: #f0f0f5;">
    <div class="flex flex-col h-screen">
        {nav_bar}

        <div class="flex-1 flex flex-col">
            <main class="flex-1 px-6 py-8 overflow-y-auto">
                <div class="max-w-6xl mx-auto w-full">
                    <!-- Links Grid -->
                    <div class="links-grid">
                        <!-- GitHub Card -->
                        <div class="link-card" style="padding: 0; background: none; border: none; box-shadow: none; position: relative;" id="github-card">
                            <a href="https://github.com/ani-portfolio/6_stock_portfolio_recommender" target="_blank" style="display: block; width: 100%; height: 100%; position: relative;">
                                <div style="width: 100%; height: 300px; background: #f3f4f6; border-radius: 20px; display: flex; align-items: center; justify-content: center; color: #6b7280;">
                                    Loading GitHub image...
                                </div>
                                <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: rgba(255, 255, 255, 0.9); color: black; padding: 12px 24px; border-radius: 8px; font-weight: 500; font-size: 1.5rem; backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); box-shadow: 0 8px 32px rgba(255, 255, 255, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.3), inset 0 -1px 0 rgba(0, 0, 0, 0.1);">
                                    GITHUB
                                </div>
                            </a>
                        </div>

                        <!-- Portfolio Card -->
                        <div class="link-card" style="padding: 0; background: none; border: none; box-shadow: none; position: relative;" id="portfolio-card">
                            <a href="https://www.datascienceportfol.io/ani_dharmarajan" target="_blank" style="display: block; width: 100%; height: 100%; position: relative;">
                                <div style="width: 100%; height: 300px; background: #f3f4f6; border-radius: 20px; display: flex; align-items: center; justify-content: center; color: #6b7280;">
                                    Loading Portfolio image...
                                </div>
                                <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: rgba(255, 255, 255, 0.9); color: black; padding: 12px 24px; border-radius: 8px; font-weight: 500; font-size: 1.5rem; backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); box-shadow: 0 8px 32px rgba(255, 255, 255, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.3), inset 0 -1px 0 rgba(0, 0, 0, 0.1);">
                                    PORTFOLIO
                                </div>
                            </a>
                        </div>

                        <!-- LinkedIn Card -->
                        <div class="link-card" style="padding: 0; background: none; border: none; box-shadow: none; position: relative;" id="linkedin-card">
                            <a href="https://www.linkedin.com/in/ani-dharmarajan/?originalSubdomain=ca" target="_blank" style="display: block; width: 100%; height: 100%; position: relative;">
                                <div style="width: 100%; height: 300px; background: #f3f4f6; border-radius: 20px; display: flex; align-items: center; justify-content: center; color: #6b7280;">
                                    Loading LinkedIn image...
                                </div>
                                <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: rgba(255, 255, 255, 0.9); color: black; padding: 12px 24px; border-radius: 8px; font-weight: 500; font-size: 1.5rem; backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); box-shadow: 0 8px 32px rgba(255, 255, 255, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.3), inset 0 -1px 0 rgba(0, 0, 0, 0.1);">
                                    LINKEDIN
                                </div>
                            </a>
                        </div>
                    </div>
                    <div class="creator-footer text-center">
                        <p class="text-gray-600 text-lg">Created by <span class="font-semibold text-gray-800">Ani Dharmarajan</span></p>
                    </div>
                </div>
            </main>
        </div>
    </div>

    <script>
        // Set active navigation
        document.getElementById('nav-links').classList.add('active');

        // Load link images using the same pattern as architecture image
        async function loadLinkImages() {{
            try {{
                const response = await fetch('/api/check-link-images');
                const images = await response.json();
                
                // Load GitHub image
                if (images.github) {{
                    document.getElementById('github-card').innerHTML = `
                        <a href="https://github.com/ani-portfolio/6_stock_portfolio_recommender" target="_blank" style="display: block; width: 100%; height: 100%; position: relative;">
                            <img src="${{images.github}}" alt="GitHub - Stock Recommender System" class="link-tile-image" style="width: 100%; height: 300px; object-fit: cover; border-radius: 20px;">
                            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: rgba(255, 255, 255, 0.9); color: black; padding: 12px 24px; border-radius: 8px; font-weight: 500; font-size: 1.5rem; backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); box-shadow: 0 8px 32px rgba(255, 255, 255, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.3), inset 0 -1px 0 rgba(0, 0, 0, 0.1);">
                                GITHUB
                            </div>
                        </a>
                    `;
                }} else {{
                    document.getElementById('github-card').innerHTML = `
                        <div style="width: 100%; height: 300px; background: #fee2e2; border-radius: 20px; display: flex; flex-direction: column; align-items: center; justify-content: center; color: #dc2626; text-align: center; padding: 20px; position: relative;">
                            <div style="font-size: 2rem; margin-bottom: 10px;">📂</div>
                            <div style="font-weight: bold; margin-bottom: 5px;">GitHub Image Not Found</div>
                            <div style="font-size: 0.9rem;">Place github image in media/ folder</div>
                            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: rgba(0, 0, 0, 0.8); color: white; padding: 12px 24px; border-radius: 8px; font-weight: 600; font-size: 1.5rem;">
                                GitHub
                            </div>
                        </div>
                    `;
                }}
                
                // Load Portfolio image
                if (images.portfolio) {{
                    document.getElementById('portfolio-card').innerHTML = `
                        <a href="https://www.datascienceportfol.io/ani_dharmarajan" target="_blank" style="display: block; width: 100%; height: 100%; position: relative;">
                            <img src="${{images.portfolio}}" alt="Portfolio - View My Portfolio" class="link-tile-image" style="width: 100%; height: 300px; object-fit: cover; border-radius: 20px;">
                            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: rgba(255, 255, 255, 0.9); color: black; padding: 12px 24px; border-radius: 8px; font-weight: 500; font-size: 1.5rem; backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); box-shadow: 0 8px 32px rgba(255, 255, 255, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.3), inset 0 -1px 0 rgba(0, 0, 0, 0.1);">
                                PORTFOLIO
                            </div>
                        </a>
                    `;
                }} else {{
                    document.getElementById('portfolio-card').innerHTML = `
                        <div style="width: 100%; height: 300px; background: #fee2e2; border-radius: 20px; display: flex; flex-direction: column; align-items: center; justify-content: center; color: #dc2626; text-align: center; padding: 20px; position: relative;">
                            <div style="font-size: 2rem; margin-bottom: 10px;">🌟</div>
                            <div style="font-weight: bold; margin-bottom: 5px;">Portfolio Image Not Found</div>
                            <div style="font-size: 0.9rem;">Place portfolio image in media/ folder</div>
                            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: rgba(0, 0, 0, 0.8); color: white; padding: 12px 24px; border-radius: 8px; font-weight: 600; font-size: 1.5rem;">
                                Portfolio
                            </div>
                        </div>
                    `;
                }}
                
                // Load LinkedIn image
                if (images.linkedin) {{
                    document.getElementById('linkedin-card').innerHTML = `
                        <a href="https://www.linkedin.com/in/ani-dharmarajan/?originalSubdomain=ca" target="_blank" style="display: block; width: 100%; height: 100%; position: relative;">
                            <img src="${{images.linkedin}}" alt="LinkedIn - Connect with me" class="link-tile-image" style="width: 100%; height: 300px; object-fit: cover; border-radius: 20px;">
                            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: rgba(255, 255, 255, 0.9); color: black; padding: 12px 24px; border-radius: 8px; font-weight: 500; font-size: 1.5rem; backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); box-shadow: 0 8px 32px rgba(255, 255, 255, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.3), inset 0 -1px 0 rgba(0, 0, 0, 0.1);">
                                LINKEDIN
                            </div>
                        </a>
                    `;
                }} else {{
                    document.getElementById('linkedin-card').innerHTML = `
                        <div style="width: 100%; height: 300px; background: #fee2e2; border-radius: 20px; display: flex; flex-direction: column; align-items: center; justify-content: center; color: #dc2626; text-align: center; padding: 20px; position: relative;">
                            <div style="font-size: 2rem; margin-bottom: 10px;">🤝</div>
                            <div style="font-weight: bold; margin-bottom: 5px;">LinkedIn Image Not Found</div>
                            <div style="font-size: 0.9rem;">Place linkedin image in media/ folder</div>
                            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: rgba(0, 0, 0, 0.8); color: white; padding: 12px 24px; border-radius: 8px; font-weight: 600; font-size: 1.5rem;">
                                LinkedIn
                            </div>
                        </div>
                    `;
                }}
                
            }} catch (error) {{
                console.error('Error loading link images:', error);
            }}
        }}

        // Load images when page loads
        document.addEventListener('DOMContentLoaded', loadLinkImages);
    </script>
</body>
</html>"""

    # Write template files
    with open("templates/index.html", "w", encoding="utf-8") as f:
        f.write(index_html)
    
    with open("templates/risk_level.html", "w", encoding="utf-8") as f:
        f.write(risk_level_html)
    
    with open("templates/links.html", "w", encoding="utf-8") as f:
        f.write(links_html)
    
    with open("templates/documentation.html", "w", encoding="utf-8") as f:
        f.write(documentation_html)
    
    with open("templates/architecture.html", "w", encoding="utf-8") as f:
        f.write(architecture_html)
    
    print("✅ Template files created successfully")

if __name__ == "__main__":
    import uvicorn
    
    # Create template files
    create_template_files()
    
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