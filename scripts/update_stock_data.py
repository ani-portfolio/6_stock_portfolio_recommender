from prefect import flow, task, get_run_logger
from prefect.blocks.system import Secret
from google.oauth2 import service_account
import json
from datetime import datetime

from google.cloud import bigquery

from functions.data_ingestion import *
from parameters import project_id, dataset_id, table_id, period

@task
def setup_gcp_credentials():
    """Load GCP credentials from Prefect Secret and return credentials object"""
    logger = get_run_logger()
    
    try:
        # Load credentials from Prefect Secret block
        secret_block = Secret.load("recommendation-app-gcp-sa-key-dev")
        credentials_json = secret_block.get()
        
        # # Parse the JSON string
        # credentials_dict = json.loads(credentials_json)
        
        # Create credentials object for BigQuery
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
def update_stock_data(credentials):
    """Update stock data with explicit credentials"""
    # Fetch base data from the URL
    df_sp500 = get_base_data('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df_sp500 = df_sp500.head()

    # Calculate metrics
    for ticker in df_sp500['Ticker'].unique().tolist():
        df_sp500 = calculate_annualized_returns(ticker_symbol=ticker, df_input=df_sp500, period=period)
    print(f"Calculated annualized returns for {len(df_sp500['Ticker'].unique().tolist())} tickers.")

    df_sp500 = df_sp500.sort_values('Market_Cap', ascending=False).reset_index(drop=True)

    # Add a new column with the current date
    df_sp500['Update_Date'] = datetime.now().strftime('%Y-%m-%d')

    client = bigquery.Client(credentials=credentials, project=project_id)
    
    # Use the client to save data
    table_id_full = f"{project_id}.{dataset_id}.{table_id}"
    job = client.load_table_from_dataframe(
        df_sp500, 
        table_id_full,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    )
    job.result()  # Wait for the job to complete

    return f"Successfully Updated Stock Data in BigQuery {datetime.now().strftime('%Y-%m-%d')}"

@flow(log_prints=True, name="update-bigquery")
def main():
    # Load GCP credentials
    credentials = setup_gcp_credentials()
    
    # Run the update task with credentials
    update_stock_data(credentials=credentials)

    print("Stock data update completed successfully.")