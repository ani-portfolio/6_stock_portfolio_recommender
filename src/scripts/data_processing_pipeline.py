from prefect import flow, task, get_run_logger
from prefect.blocks.system import Secret
from google.oauth2 import service_account
from datetime import datetime
import pandas as pd

from google.cloud import bigquery
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from utils.data_ingestion import *
from parameters import *

@task
def setup_gcp_credentials(service_account_key):
    """Load GCP credentials from Prefect Secret and return credentials object"""
    logger = get_run_logger()
    
    try:
        secret_block = Secret.load(service_account_key)
        credentials_json = secret_block.get()
        
        credentials = service_account.Credentials.from_service_account_info(
            credentials_json,
            scopes=["https://www.googleapis.com/auth/bigquery"]
        )
        
        logger.info("GCP credentials loaded successfully from Prefect Secret")
        return credentials

    except Exception as e:
        logger.error(f"Failed to load GCP credentials: {str(e)}")
        raise

@task
def connect_to_pinecone(pinecone_api_key):
    """Initialize Pinecone connection and return client"""  

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
    """Initialize HuggingFace embedding model"""
    
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
    """Create text chunks from stock data for embedding"""
    
    try:
        documents = []
        columns = df.columns.tolist()
        
        for _, row in df.iterrows():
            doc_content = ""
            for col in columns:
                if pd.notna(row[col]):
                    doc_content += f"{col}: {row[col]}\n"
            
            metadata = {
                "Ticker": row['Ticker'],
                "Company_Name": row['Company_Name'], 
                "Sector": row['Sector'],
                "Industry": row['Industry'],
            }

            documents.append(Document(page_content=doc_content.strip(), metadata=metadata))
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]  
        )
        
        chunks = text_splitter.split_documents(documents)

        print(f"Created {len(chunks)} chunks from {len(df)} stock records")
        return chunks

    except Exception as e:
        print(f"Error creating text chunks: {str(e)}")
        raise

@task
def create_embeddings_with_model(chunks, embeddings_model):
    """Create embeddings for text chunks using the embedding model"""
    
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
    """Save vector embeddings to Pinecone with metadata and page content"""
    
    try:
        existing_indexes = pc.list_indexes().names()
        
        # Clear existing data if requested (recommended for stock data)
        if index_name in existing_indexes and clear_existing:
            print("Clearing existing data from index...")
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
def get_stock_data(url):
    """Update stock data with explicit credentials"""
    df_sp500 = get_base_data(url)

    for ticker in df_sp500['Ticker'].unique().tolist():
        df_sp500 = calculate_annualized_returns(ticker_symbol=ticker, df_input=df_sp500, period=period)
    print(f"Calculated annualized returns for {len(df_sp500['Ticker'].unique().tolist())} tickers.")

    df_sp500 = df_sp500.sort_values('Market_Cap', ascending=False).reset_index(drop=True)

    df_sp500['Update_Date'] = datetime.now().strftime('%Y-%m-%d')

    return df_sp500

@task
def save_to_bigquery(df, credentials):
    client = bigquery.Client(credentials=credentials, project=project_id)
    
    table_id_full = f"{project_id}.{dataset_id}.{table_id}"
    job = client.load_table_from_dataframe(
        df, 
        table_id_full,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    )
    job.result()

    return f"Successfully Updated Stock Data in BigQuery {datetime.now().strftime('%Y-%m-%d')}"

@flow(log_prints=True, name="data-processing-pipeline")
def data_processing_flow():

    print("Loading GCP credentials")
    credentials = setup_gcp_credentials(prefect_gcp_service_account_key)

    print("Connecting to Pinecone")
    pc = connect_to_pinecone(prefect_pinecone_api_key)
    
    print("Connecting to HuggingFace Embeddings")
    embeddings_model = connect_to_huggingface_embeddings(huggingface_embeddings_model)

    print("Fetching stock data")
    df_sp500 = get_stock_data(base_data_url)
    
    print("Saving stock data to BigQuery")
    bigquery_result = save_to_bigquery(df_sp500, credentials)
    
    print("Creating text chunks from stock data")
    chunks = create_text_chunks(df_sp500)
    
    print("Creating embeddings using HuggingFace Embeddings")
    embeddings = create_embeddings_with_model(chunks, embeddings_model)
    
    print("Saving embeddings to Pinecone")
    pinecone_result = save_embeddings_to_pinecone(pc, chunks, embeddings, pinecone_index_name, clear_existing)

    print("Enhanced Data Processing Pipeline Completed Successfully")
    print(f"BigQuery: {bigquery_result}")
    print(f"Pinecone: {pinecone_result}")

if __name__ == "__main__":
    data_processing_flow()