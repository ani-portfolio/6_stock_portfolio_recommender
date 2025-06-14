from prefect import flow, task
from datetime import datetime

from functions.data_ingestion import *
from parameters import dataset_id, table_id, period

@task
def update_stock_data():
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

    # Save the DataFrame to BigQuery
    save_table_to_bigquery(df_sp500, dataset_id=dataset_id, table_id=table_id)

    return f"Successfully Update Stock Data in BigQuery {datetime.now().strftime('%Y-%m-%d')}"

@flow(log_prints=True, name="update-bigquery")
def main():
    update_stock_data()

if __name__ == "__main__":
    # Deploy without from_source - let the work pool handle code execution
    main.deploy(
        name="update-stock-data-bigquery",
        work_pool_name="default-work-pool",
        cron="0 */6 * * *",  # Run every 6 hours
        tags=["stock-data", "bigquery"],
        description="Updates stock data in BigQuery every 6 hours",
        image="prefecthq/prefect:2-python3.11",  # Specify a base image
        build=False,  # Don't build a new image
    )