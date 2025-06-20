

# GCP Parameters
staging_dataset = "stock_data"
location = "US"
project_id="capable-arbor-293714"
dataset_id = 'stock_data'
table_id = 'stock_data'

# Data Processing Parameters
base_data_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
period = "5y"

# Secret Key Name 
prefect_gcp_service_account_key = "recommendation-app-gcp-sa-key-dev"  # Prefect Secret name for GCP credentials
prefect_pinecone_api_key = "recommendation-app-pinecone-api-key"

# Pinecone Parameters
huggingface_embeddings_model = "sentence-transformers/all-MiniLM-L6-v2"
pinecone_index_name = "stock-recommendation-app-index"
clear_existing = True

# Groq Parameters
groq_llm_model = "llama-3.3-70b-versatile"
