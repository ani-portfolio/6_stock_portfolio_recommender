# deploy_stock_update.py
from prefect import flow, task

from datetime import datetime
from google.cloud import bigquery

from functions.data_ingestion import *
from parameters import dataset_id, table_id, period

@task
def update_stock_data():
    # Fetch base data from the URL
    df_sp500 = get_base_data('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df_sp500 = df_sp500.head(20)

    # Calculate metrics
    for tickers in df_sp500['Ticker'].unique().tolist():
        df_sp500 = calculate_annualized_returns(ticker_symbol=tickers, df_input=df_sp500, period=period)
    print(f"Calculated annualized returns for {len(df_sp500['Ticker'].unique().tolist())} tickers.")

    df_sp500 = df_sp500.sort_values('Market_Cap', ascending=False).reset_index(drop=True)

    # Add a new column with the current date
    df_sp500['Update_Date'] = datetime.now().strftime('%Y-%m-%d')

    # Save the DataFrame to BigQuery
    client = bigquery.Client()
    save_table_to_bigquery(df_sp500, dataset_id=dataset_id, table_id=table_id)

    return f"Successfully Update Stock Data in BigQuery {datetime.now().strftime('%Y-%m-%d')}"

@flow(log_prints=True)
def main():
    update_stock_data()

if __name__ == "__main__":
    main.deploy(
        name="update-stock-data-bigquery",
        work_pool_name="default-work-pool",
        image="anizehrs/ani-docker:latest",
    )