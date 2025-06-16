

# BigQuery parameters
STAGING_DATASET = "stock_data"
LOCATION = "US"
URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
PROJECT_ID="capable-arbor-293714"
dataset_id = 'stock_data'
table_id = 'stock_data_updated_via_github_actions'
period = "5y"

# Streamliit local app parameters
PERSIST_DIRECTORY = "/Users/ani/Projects/6_stock_portfolio_recommendation/chroma_db" # Directory to store ChromaDB
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Sentence Transformers model
LLM_MODEL = "mistral" # Ensure this model is pulled in Ollama (ollama pull mistral)