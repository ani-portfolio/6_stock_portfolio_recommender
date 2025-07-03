from prefect import flow, task, get_run_logger
from prefect.blocks.system import Secret
from google.oauth2 import service_account
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from google.cloud import bigquery
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

from data_ingestion import *
from parameters import *
from sentiment import *

# @task
# def setup_gcp_credentials(service_account_key):
#     """
#     Load GCP credentials from Prefect Secret and return credentials object.
#     Args:
#         service_account_key: Prefect secret key for GCP service account
#     Returns:
#         GCP credentials object
#     """
#     logger = get_run_logger()
    
#     try:
#         secret_block = Secret.load(service_account_key)
#         credentials_json = secret_block.get()
        
#         credentials = service_account.Credentials.from_service_account_info(
#             credentials_json,
#             scopes=["https://www.googleapis.com/auth/bigquery"]
#         )
        
#         logger.info("GCP credentials loaded successfully from Prefect Secret")
#         return credentials

#     except Exception as e:
#         logger.error(f"Failed to load GCP credentials: {str(e)}")
#         raise

@task
def connect_to_pinecone(pinecone_api_key):
    """
    Initialize Pinecone connection and return client.
    Args:
        pinecone_api_key: Prefect secret key for Pinecone API
    Returns:
        Pinecone client object
    """
    logger = get_run_logger()

    try:        
        secret_block = Secret.load(pinecone_api_key)
        pinecone_api = secret_block.get()
        pc = Pinecone(api_key=pinecone_api)

        logger.info("Pinecone client initialized successfully")
        return pc

    except Exception as e:
        logger.error(f"Failed to initialize Pinecone client: {str(e)}")
        raise

@task
def connect_to_huggingface_embeddings(huggingface_embeddings_model):
    """
    Initialize HuggingFace embedding model.
    Args:
        huggingface_embeddings_model: Model name for HuggingFace embeddings
    Returns:
        HuggingFace embeddings object
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=huggingface_embeddings_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        return embeddings

    except Exception as e:
        raise

@task
def create_text_chunks(df):
    """
    Create exactly one text chunk per stock row.
    Args:
        df: DataFrame containing stock data
    Returns:
        List of Document objects with stock information
    """
    try:
        chunks = []
        
        for _, row in df.iterrows():
            doc_content = ""
            for col in df.columns:
                if pd.notna(row[col]):
                    doc_content += f"{col}: {row[col]}\n"
            
            metadata = {
                "Ticker": row['Ticker'],
                "Company_Name": row['Company_Name'], 
                "Sector": row['Sector'] if pd.notna(row['Sector']) else None,
                "Industry": row['Industry'] if pd.notna(row['Industry']) else None,
            }

            chunk = Document(page_content=doc_content.strip(), metadata=metadata)
            chunks.append(chunk)
        
        print(f"Created {len(chunks)} chunks from {len(df)} stock records (1:1 ratio)")
        return chunks

    except Exception as e:
        print(f"Error creating text chunks: {str(e)}")
        raise

@task
def create_embeddings_with_model(chunks, embeddings_model):
    """
    Create embeddings for text chunks using the embedding model.
    Args:
        chunks: List of Document objects
        embeddings_model: HuggingFace embeddings model
    Returns:
        List of embeddings vectors
    """
    try:
        texts = [chunk.page_content for chunk in chunks]
        embeddings = embeddings_model.embed_documents(texts)

        print(f"Successfully created embeddings for {len(texts)} chunks")
        return embeddings
        
    except Exception as e:
        print(f"Failed to create embeddings: {str(e)}")
        raise

@task
def save_embeddings_to_pinecone(pc, chunks, embeddings, index_name, clear_existing):
    """
    Save vector embeddings to Pinecone with metadata and page content.
    Args:
        pc: Pinecone client object
        chunks: List of Document objects
        embeddings: List of embedding vectors
        index_name: Name of Pinecone index
        clear_existing: Boolean to clear existing data
    Returns:
        Success message string
    """
    try:
        existing_indexes = pc.list_indexes().names()
        
        if index_name in existing_indexes and clear_existing:
            print("Clearing existing data from index...")
            index = pc.Index(index_name)
            index.delete(delete_all=True)
            print("Index cleared successfully")

        if index_name not in existing_indexes:
            print(f"Creating new Pinecone index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=len(embeddings[0]),
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            print("Index created successfully")
        
        index = pc.Index(index_name)

        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_id = f"{chunk.metadata.get('Ticker', 'unknown')}_{i}_{chunk.metadata.get('Update_Date', '')}"
            
            metadata = {
                'Ticker': chunk.metadata.get('Ticker'),
                'Company_Name': chunk.metadata.get('Company_Name'),
                'Sector': chunk.metadata.get('Sector'),
                'Industry': chunk.metadata.get('Industry'),
                'content': chunk.page_content,
                'chunk_index': i
            }
            
            vectors.append({
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            })
        
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
            print(f"Upserted batch {i//batch_size + 1}/{(len(vectors) + batch_size - 1)//batch_size}")

        print(f"Successfully saved {len(vectors)} embeddings to Pinecone index '{index_name}'")
        return f"Saved {len(vectors)} embeddings to Pinecone"
        
    except Exception as e:
        print(f"Failed to save embeddings to Pinecone: {str(e)}")
        raise

@task
def get_multiple_stocks_data(tickers):
    """
    Get stock data for multiple tickers and return as DataFrame.
    Args:
        tickers: List of ticker symbols
        period: Historical data period
    Returns:
        DataFrame with stock data for all tickers
    """
    all_data = []

    for ticker in (tickers):
        try:
            stock_data = get_stock_data(ticker)
            if stock_data is not None:
                tqdm.write(f'Processing {ticker}')
                all_data.append(stock_data)
            else:
                tqdm.write(f"Warning: No data retrieved for {ticker}")
        except Exception as e:
            tqdm.write(f"Error processing {ticker}: {str(e)}")
            continue
            
    df = pd.DataFrame(all_data)

    df['Dividend_Yield'] = df['Dividend_Yield'].fillna(0)
    df['Sector'] = df['Sector'].fillna('Unknown')
    df['Industry'] = df['Industry'].fillna('Unknown') 
    df['Country'] = df['Country'].fillna('Unknown')
    df['Business_Summary'] = df['Business_Summary'].fillna('No description available')
    df = df.fillna(0)

    today_date = datetime.now().date().strftime("%Y-%m-%d")
    df['Update_Date'] = today_date

    return df.sort_values('Market_Cap', ascending=False, na_position='last')

@task
def save_to_bigquery(df, credentials):
    """
    Save DataFrame to BigQuery table.
    Args:
        df: DataFrame to save
        credentials: GCP credentials object
    Returns:
        Success message string
    """
    client = bigquery.Client(credentials=credentials, project=project_id)
    
    table_id_full = f"{project_id}.{dataset_id}.{table_id}"
    job = client.load_table_from_dataframe(
        df, 
        table_id_full,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    )
    job.result()

    return f"Successfully Updated Stock Data in BigQuery {datetime.now().strftime('%Y-%m-%d')}"

@task
def sentiment_analysis(df, api_key, base_url):
    """
    Perform sentiment analysis on stock data.
    Args:
        df: DataFrame containing stock data
    Returns:
        DataFrame with sentiment analysis results added
    """
    tickers = df['Ticker'].unique().tolist()
    df_sentiment = analyze_multiple_stocks(tickers, api_key, base_url)
    df = pd.merge(df, df_sentiment, how='left', on='Ticker')
    return df.sort_values(['Market_Cap'], ascending=False).reset_index(drop=True)

@flow(log_prints=True, name="data-processing-pipeline")
def data_processing_flow():
    """
    Main data processing flow for stock data pipeline.
    Orchestrates the entire ETL process from data ingestion to vector storage.
    """
    # print("Loading GCP credentials")
    # credentials = setup_gcp_credentials(prefect_gcp_service_account_key)

    print("Connecting to Pinecone")
    pc = connect_to_pinecone(prefect_pinecone_api_key)
    
    print("Connecting to HuggingFace Embeddings")
    embeddings_model = connect_to_huggingface_embeddings(huggingface_embeddings_model)

    print("Fetching stock tickers")
    stock_tickers = get_tickers(base_data_url)

    print("Create Dataset")
    df_enriched_stock_data = get_multiple_stocks_data(stock_tickers)

    print("Add Sentiment")
    df_enriched_stock_data = sentiment_analysis(df_enriched_stock_data, Secret.load(prefect_newsapi_key).get(), sentiment_base_url)
    
    print("Saving stock data to BigQuery")
    bigquery_result = save_to_bigquery(df_enriched_stock_data, credentials)
    
    print("Creating text chunks from stock data")
    chunks = create_text_chunks(df_enriched_stock_data)
    
    print("Creating embeddings using HuggingFace Embeddings")
    embeddings = create_embeddings_with_model(chunks, embeddings_model)
    
    print("Saving embeddings to Pinecone")
    pinecone_result = save_embeddings_to_pinecone(pc, chunks, embeddings, pinecone_index_name, clear_existing)

    print("Enhanced Data Processing Pipeline Completed Successfully")
    print(f"BigQuery: {bigquery_result}")
    print(f"Pinecone: {pinecone_result}")

if __name__ == "__main__":
    data_processing_flow()