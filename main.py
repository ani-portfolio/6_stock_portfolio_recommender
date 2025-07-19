from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import sys
import asyncio
from typing import Optional

# Add your src path
sys.path.append('..')
try:
    from src.parameters import *
    from src.rag import *
    from src.query_router import *
    from src.data_ingestion import *
    from src.recommender import *
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    # Define fallback functions for development
    def router(query):
        return {'route': 'semantic_search'}
    
    def rag_query_stocks(**kwargs):
        return {'success': True, 'answer': 'Mock response for development'}
    
    def recommend_stocks_from_query(**kwargs):
        return {'success': True, 'answer': 'Mock recommendation response'}
    
    def load_table_from_bigquery(*args):
        return None
    
    def create_sample_queries():
        pass

app = FastAPI(title="Stock Recommendation System", version="1.0.0")

# Create directories
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

templates = Jinja2Templates(directory="templates")

# Sample queries for the UI
SAMPLE_QUERIES = [
    "Which tech stocks have high returns and low risk?",
    "Show me dividend stocks with strong fundamentals",
    "What are the best performing healthcare stocks?",
    "Find stocks with low volatility and steady growth",
    "Which energy stocks are undervalued right now?",
    "Show me emerging market opportunities"
]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main stock recommendation page"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "sample_queries": SAMPLE_QUERIES
    })

@app.post("/analyze")
async def analyze_stock_query(
    request: Request,
    user_query: str = Form(...),
    analysis_type: Optional[str] = Form("auto")
):
    """Analyze stock query and return recommendations"""
    
    if not user_query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Route the query
        result = router(user_query)
        
        response_data = {
            "query": user_query,
            "route": result.get('route', 'unknown'),
            "success": False,
            "answer": "",
            "error": None
        }
        
        if result['route'] == 'semantic_search':
            try:
                # Get configuration from parameters
                index_name = globals().get('pinecone_index_name', 'stock-index')
                groq_model = globals().get('groq_llm_model', 'mixtral-8x7b-32768')
                embeddings_model = globals().get('huggingface_embeddings_model', 'all-MiniLM-L6-v2')
                
                response = rag_query_stocks(
                    query=user_query,
                    top_k=10,
                    groq_llm_model=groq_model,
                    huggingface_embeddings_model=embeddings_model,
                    pinecone_index_name=index_name
                )
                response_data.update(response)
                
            except Exception as e:
                response_data["error"] = f"Semantic search failed: {str(e)}"
                
        elif result['route'] == 'recommender':
            try:
                # Load data from BigQuery
                dataset_id = globals().get('dataset_id', 'your_dataset')
                table_id = globals().get('table_id', 'your_table')
                project_id = globals().get('project_id', 'your_project')
                
                df_stock_data = load_table_from_bigquery(dataset_id, table_id, project_id)
                
                if df_stock_data is not None:
                    non_numerical_columns = [
                        'Ticker', 'Company_Name', 'Sector', 'Industry', 
                        'Country', 'Business_Summary', 'Sentiment', 'Update_Date'
                    ]
                    numerical_columns = df_stock_data.drop(non_numerical_columns, axis=1).columns.tolist()
                    
                    groq_api_key = globals().get('get_api_key', lambda x: 'your_api_key')("GROQ_API_KEY")
                    
                    response = recommend_stocks_from_query(
                        df=df_stock_data,
                        user_query=user_query,
                        numerical_columns=numerical_columns,
                        non_numerical_columns=non_numerical_columns,
                        top_n=5,
                        groq_api_key=groq_api_key
                    )
                    response_data.update(response)
                else:
                    response_data["error"] = "Could not load stock data"
                    
            except Exception as e:
                response_data["error"] = f"Recommendation failed: {str(e)}"
        
        else:
            response_data["error"] = f"Unknown route: {result.get('route', 'none')}"
            
    except Exception as e:
        response_data["error"] = f"Analysis failed: {str(e)}"
    
    return JSONResponse(content=response_data)

@app.get("/api/sample-queries")
async def get_sample_queries():
    """Get sample queries for the frontend"""
    return {"queries": SAMPLE_QUERIES}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Stock Recommendation System"}

def create_template_files():
    """Create HTML template files"""
    
    # Main template
    index_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Recommendation System</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    animation: {
                        'fade-in': 'fadeIn 0.5s ease-in-out',
                        'slide-up': 'slideUp 0.6s ease-out',
                        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                        'spin-slow': 'spin 3s linear infinite',
                    }
                }
            }
        }
    </script>
    <style>
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .glass {
            background: rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .glass-light {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.3);
        }
        .loading {
            display: none;
        }
        .loading.active {
            display: flex;
        }
    </style>
</head>
<body class="min-h-screen bg-gradient-to-br from-red-600 via-black to-blue-800">
    <!-- Header -->
    <header class="glass border-b border-red-400/30 sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
            <div class="flex items-center justify-between">
                <div class="flex items-center space-x-3">
                    <div class="w-10 h-10 bg-gradient-to-r from-red-500 to-blue-500 rounded-lg flex items-center justify-center">
                        <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path>
                        </svg>
                    </div>
                    <h1 class="text-2xl font-bold text-white">Stock Recommendation System</h1>
                </div>
                <div class="flex items-center space-x-2 text-blue-200 text-sm">
                    <span>Powered by</span>
                    <span class="text-red-400 font-semibold">RAG</span>
                    <span>•</span>
                    <span class="text-blue-400 font-semibold">Pinecone</span>
                    <span>•</span>
                    <span class="text-white font-semibold">Groq</span>
                </div>
            </div>
        </div>
    </header>

    <!-- Main Content -->
    <main class="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <!-- Search Section -->
        <div class="text-center mb-12 animate-fade-in">
            <h2 class="text-4xl md:text-5xl font-bold text-white mb-4">
                AI-Powered Stock Analysis
            </h2>
            <p class="text-xl text-blue-200 max-w-3xl mx-auto mb-8">
                Get intelligent stock recommendations and market insights powered by advanced AI
            </p>
            
            <!-- Search Form -->
            <div class="glass-light rounded-2xl p-8 max-w-4xl mx-auto border border-red-400/30">
                <form id="analysisForm" class="space-y-6">
                    <div class="flex flex-col md:flex-row gap-4">
                        <div class="flex-1">
                            <textarea 
                                id="userQuery"
                                name="user_query"
                                placeholder="🔍 Ask about stocks, sectors, or portfolio strategies..."
                                rows="3"
                                required
                                class="w-full px-4 py-3 rounded-lg bg-black/30 border border-red-400/40 text-white placeholder-blue-300/60 focus:outline-none focus:ring-2 focus:ring-red-400/60 focus:border-red-400 transition-all duration-300 backdrop-blur-sm resize-none"
                            ></textarea>
                        </div>
                        <div class="flex flex-col justify-end">
                            <button 
                                type="submit"
                                id="analyzeBtn"
                                class="bg-gradient-to-r from-red-600 to-blue-600 hover:from-red-700 hover:to-blue-700 text-white font-semibold py-3 px-8 rounded-lg transition-all duration-300 flex items-center justify-center space-x-2 border border-red-400/50 hover:border-red-400 hover:scale-105 transform shadow-lg min-w-[120px] h-[52px]"
                            >
                                <span id="btnText">🚀 Analyze</span>
                                <div id="btnSpinner" class="loading items-center">
                                    <svg class="animate-spin h-5 w-5 text-white" viewBox="0 0 24 24">
                                        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"></circle>
                                        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                    </svg>
                                </div>
                            </button>
                        </div>
                    </div>
                </form>
            </div>
        </div>

        <!-- Sample Queries -->
        <div class="mb-12 animate-slide-up">
            <div class="glass-light rounded-xl p-6 border border-blue-400/30">
                <h3 class="text-xl font-semibold text-white mb-4 text-center">Try these sample queries:</h3>
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                    {% for query in sample_queries %}
                    <button 
                        onclick="setQuery('{{ query }}')"
                        class="text-left bg-black/20 hover:bg-red-600/30 text-blue-200 hover:text-white p-3 rounded-lg transition-all duration-300 border border-white/20 hover:border-red-400/50 text-sm"
                    >
                        {{ query }}
                    </button>
                    {% endfor %}
                </div>
            </div>
        </div>

        <!-- Results Section -->
        <div id="resultsSection" class="hidden animate-slide-up">
            <div class="glass-light rounded-2xl p-8 border border-blue-400/30">
                <div class="flex items-center justify-between mb-6">
                    <h3 class="text-2xl font-bold text-white">Analysis Results</h3>
                    <span id="routeIndicator" class="px-3 py-1 rounded-full text-xs font-medium"></span>
                </div>
                
                <div id="resultsContent" class="prose prose-invert max-w-none">
                    <!-- Results will be inserted here -->
                </div>
                
                <div class="mt-6 p-4 bg-yellow-500/20 border border-yellow-400/50 rounded-lg">
                    <div class="flex items-center space-x-2">
                        <svg class="w-5 h-5 text-yellow-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z"></path>
                        </svg>
                        <p class="text-yellow-200 text-sm font-medium">
                            ⚠️ This information is not financial advice and should not be considered as such
                        </p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Error Section -->
        <div id="errorSection" class="hidden animate-slide-up">
            <div class="glass-light rounded-2xl p-8 border border-red-500/30">
                <div class="flex items-center space-x-3 mb-4">
                    <svg class="w-6 h-6 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    </svg>
                    <h3 class="text-xl font-bold text-red-400">Analysis Failed</h3>
                </div>
                <div id="errorContent" class="text-red-200"></div>
                <p class="text-blue-200 text-sm mt-4">
                    Please check your API keys and ensure all dependencies are properly configured.
                </p>
            </div>
        </div>
    </main>

    <script>
        function setQuery(query) {
            document.getElementById('userQuery').value = query;
        }

        function showLoading() {
            document.getElementById('btnText').style.display = 'none';
            document.getElementById('btnSpinner').classList.add('active');
            document.getElementById('analyzeBtn').disabled = true;
        }

        function hideLoading() {
            document.getElementById('btnText').style.display = 'inline';
            document.getElementById('btnSpinner').classList.remove('active');
            document.getElementById('analyzeBtn').disabled = false;
        }

        function formatMarkdown(text) {
            // Simple markdown-to-HTML conversion
            return text
                .replace(/\\n/g, '<br>')
                .replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>')
                .replace(/\\*(.*?)\\*/g, '<em>$1</em>')
                .replace(/`(.*?)`/g, '<code class="bg-gray-800 px-1 rounded">$1</code>');
        }

        function showResults(data) {
            const resultsSection = document.getElementById('resultsSection');
            const resultsContent = document.getElementById('resultsContent');
            const routeIndicator = document.getElementById('routeIndicator');
            
            // Set route indicator
            const routeColors = {
                'semantic_search': 'bg-blue-500/20 text-blue-300 border border-blue-400/50',
                'recommender': 'bg-red-500/20 text-red-300 border border-red-400/50'
            };
            
            routeIndicator.className = `px-3 py-1 rounded-full text-xs font-medium ${routeColors[data.route] || 'bg-gray-500/20 text-gray-300'}`;
            routeIndicator.textContent = data.route === 'semantic_search' ? 'Semantic Search' : 'AI Recommender';
            
            // Set content
            resultsContent.innerHTML = `
                <div class="text-white space-y-4">
                    <div class="bg-black/20 p-4 rounded-lg border border-white/10">
                        <h4 class="font-semibold text-blue-200 mb-2">Query:</h4>
                        <p class="text-white">${data.query}</p>
                    </div>
                    <div class="bg-black/20 p-4 rounded-lg border border-white/10">
                        <h4 class="font-semibold text-blue-200 mb-2">Analysis:</h4>
                        <div class="text-white leading-relaxed">${formatMarkdown(data.answer)}</div>
                    </div>
                </div>
            `;
            
            resultsSection.classList.remove('hidden');
            document.getElementById('errorSection').classList.add('hidden');
        }

        function showError(error) {
            const errorSection = document.getElementById('errorSection');
            const errorContent = document.getElementById('errorContent');
            
            errorContent.textContent = error;
            errorSection.classList.remove('hidden');
            document.getElementById('resultsSection').classList.add('hidden');
        }

        document.getElementById('analysisForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const userQuery = document.getElementById('userQuery').value.trim();
            if (!userQuery) {
                alert('Please enter a query');
                return;
            }
            
            showLoading();
            
            try {
                const formData = new FormData();
                formData.append('user_query', userQuery);
                
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    showResults(data);
                } else {
                    showError(data.error || 'Analysis failed');
                }
                
            } catch (error) {
                showError(`Network error: ${error.message}`);
            } finally {
                hideLoading();
            }
        });
    </script>
</body>
</html>'''
    
    with open("templates/index.html", "w") as f:
        f.write(index_html)

if __name__ == "__main__":
    import uvicorn
    
    # Create template files
    create_template_files()
    
    # Initialize sample queries
    try:
        create_sample_queries()
    except:
        pass
    
    print("🚀 Starting Stock Recommendation System...")
    print("📊 Features: RAG • Pinecone • Groq • FastAPI • Tailwind CSS")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)