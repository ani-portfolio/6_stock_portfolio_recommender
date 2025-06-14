# deploy_stock_update.py
from prefect import flow, task
from prefect.deployments import Deployment
from prefect.runner.storage import GitRepository

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

@flow(name="stock-portfolio-update")
def main():
    result = update_stock_data()
    print(result)
    return result

if __name__ == "__main__":
    # Create deployment for your specific repository
    deployment = Deployment.build_from_flow(
        flow=main,
        name="stock-data-updater",
        version="1.0",
        tags=["stocks", "bigquery", "daily"],
        storage=GitRepository(
            url="https://github.com/ani-portfolio/6_stock_portfolio_recommender.git",
            branch="feature/prefect",  # Using your feature branch
        ),
        entrypoint="update_stock_data.py:main",  # Points to the main flow
        work_pool_name="default-agent-pool",  # Update this if you have a different work pool
        parameters={},  # Add any default parameters if needed
        description="Daily update of S&P 500 stock data to BigQuery"
    )
    
    # Apply the deployment
    deployment.apply()
    print("✅ Deployment 'stock-data-updater' created successfully!")
    print("📍 Repository: https://github.com/ani-portfolio/6_stock_portfolio_recommender.git")
    print("🌿 Branch: feature/prefect")
    print("📄 Entrypoint: update_stock_data.py:main")