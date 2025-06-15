

# BigQuery parameters
project_id="capable-arbor-293714"
staging_dataset = "stock_data"
location = "US"
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
dataset_id = 'stock_data'
table_id = 'stock_data_updated_via_github_actions'
period = "5y"

# Streamlit local app parameters
persist_directory = "/Users/ani/Projects/6_stock_portfolio_recommendation/chroma_db" # Directory to store ChromaDB
embedding_model = "all-MiniLM-L6-v2" # Sentence Transformers model
llm_model = "mistral" # Ensure this model is pulled in Ollama (ollama pull mistral)