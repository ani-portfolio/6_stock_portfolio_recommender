from datetime import datetime
from google.cloud import bigquery

from data_ingestion import *
from parameters import dataset_id, table_id, period

# Fetch base data from the URL
df_sp500 = get_base_data('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')

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