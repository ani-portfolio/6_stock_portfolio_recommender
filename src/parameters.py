# GCP Parameters
staging_dataset = "stock_data"
location = "US"
project_id="capable-arbor-293714"
dataset_id = 'stock_data'
table_id = 'stock_data'

# Data Processing Parameters
base_data_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
period = "10y"

# Secret Key Name 
prefect_gcp_service_account_key = "recommendation-app-gcp-sa-key-dev"
prefect_pinecone_api_key = "recommendation-app-pinecone-api-key"
prefect_newsapi_key = "recommendation-app-news-api-key"

# Pinecone Parameters
huggingface_embeddings_model = "FinLang/finance-embeddings-investopedia"
pinecone_index_name = "stock-recommendation-app-index"
clear_existing = True

# Groq Parameters
groq_llm_model = "llama-3.3-70b-versatile"

# NewsAPI
sentiment_base_url = "https://newsapi.org/v2/everything"