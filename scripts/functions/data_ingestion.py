from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_gbq

def get_base_data(url):
    """Fetches base data from a given URL and returns it as a DataFrame.
    Args:
        url (str): The URL to fetch the data from.
    Returns:
        pd.DataFrame: A DataFrame containing the fetched data.
    """
    
    df = pd.read_html(url)[0]
    df = df.rename(columns={'Symbol': 'Ticker', 'Security': 'Company_Name', 'GICS Sector': 'Sector', 'GICS Sub-Industry': 'Industry', 'Founded': 'Founded_Year'}).drop(['Date added', 'CIK'], axis=1)
    df.columns = df.columns.str.replace(' ', '_').str.replace('/', '_').str.replace('-', '_')
    df['Ticker'] = df['Ticker'].str.upper()
    
    print(f"Fetched {len(df)} rows from {url}")
    return df

def create_schema(df):
    """Creates a BigQuery schema based on the DataFrame's columns and their data types.
    Args:
        df (pd.DataFrame): The DataFrame for which to create the schema.
    Returns:
        list: A list of bigquery.SchemaField objects representing the schema.
    """

    schema = []
    for col in df.columns:
        if df[col].dtype == 'float64':
            schema.append(bigquery.SchemaField(col, 'FLOAT'))
        elif df[col].dtype == 'int64':
            schema.append(bigquery.SchemaField(col, 'INTEGER'))
        else:
            schema.append(bigquery.SchemaField(col, 'STRING'))
    
    print(f"Created schema with {len(schema)} fields.")
    return schema

def calculate_annualized_returns(ticker_symbol, df_input, period="5y"):
    """Calculate annualized returns for a given ticker over a specified period.
    Args:
        ticker (yfinance.Ticker): The ticker object for the stock.
        period (str): The period over which to calculate returns (default is "5y").
    Returns:
        pd.Series: A series of annualized returns.
    """

    try:
        ticker = yf.Ticker(ticker_symbol)

        # Get historical market data
        hist = ticker.history(period=period).reset_index()
        hist['Date'] = pd.to_datetime(hist['Date'])
        hist['Year'] = hist['Date'].dt.year
        current_year = hist['Year'].max()    

        # Pct from all time high
        all_time_high = hist['Close'].max()
        df_input.loc[df_input['Ticker'] == ticker_symbol, 'Pct_From_All_Time_High'] = np.round(((hist['Close'].iloc[-1] - all_time_high) / all_time_high) * 100, 2)
        
        # 24 Hour Change
        hist['24_Hour_Change'] = hist['Close'].pct_change(periods=1) * 100
        df_input.loc[df_input['Ticker'] == ticker_symbol, '24_Hour_Change'] = np.round(hist['24_Hour_Change'].iloc[-1], 2)

        # 7 day Change
        hist['7_Day_Change'] = hist['Close'].pct_change(periods=7) * 100
        df_input.loc[df_input['Ticker'] == ticker_symbol, '7_Day_Change'] = np.round(hist['7_Day_Change'].iloc[-1], 2)

        # 30 Day Change
        hist['30_Day_Change'] = hist['Close'].pct_change(periods=30) * 100
        df_input.loc[df_input['Ticker'] == ticker_symbol, '30_Day_Change'] = np.round(hist['30_Day_Change'].iloc[-1], 2)

        # average annualized return
        annualized_return = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) ** (1 / (current_year - hist['Year'].min())) - 1) * 100
        df_input.loc[df_input['Ticker'] == ticker_symbol, 'Annualized_Return'] = np.round(annualized_return, 2)

        # Calculate Percent Returns
        for year in hist['Year'].sort_values(ascending=False).unique():
            if year == current_year:
                ytd_return = ((hist[hist['Year'] == year]['Close'].iloc[-1] - hist[hist['Year'] == year]['Close'].iloc[0]) / hist[hist['Year'] == year]['Close'].iloc[0]) * 100
                df_input.loc[df_input['Ticker'] == ticker_symbol, 'YTD_Pct_Return'] = np.round(ytd_return, 2)
            elif year < current_year:
                annual_return = ((hist[hist['Year'] == year]['Close'].iloc[-1] - hist[hist['Year'] == year]['Close'].iloc[0]) / hist[hist['Year'] == year]['Close'].iloc[0]) * 100
                df_input.loc[df_input['Ticker'] == ticker_symbol, f'{year}_Pct_Return'] = np.round(annual_return, 2)
        
        # Get market cap
        df_input.loc[df_input['Ticker'] == ticker_symbol, 'Market_Cap'] = ticker.info.get('marketCap', np.nan)
        
        # Calculate 200 Day Moving Average & Pct Difference from it
        hist['200_MA'] = hist['Close'].rolling(window=200).mean()
        hist['Pct_Diff_200_MA'] = ((hist['Close'] - hist['200_MA']) / hist['200_MA']) * 100
        df_input.loc[df_input['Ticker'] == ticker_symbol, 'Pct_Diff_200_MA'] = np.round(hist['Pct_Diff_200_MA'].iloc[-1], 2)
        
        # Calculate Volatility
        hist['Daily_Return'] = hist['Close'].pct_change()
        mean_daily_return = hist['Daily_Return'].mean()
        volatility = (((hist['Daily_Return'] - mean_daily_return) ** 2).mean() ** 0.5) * np.sqrt(252)  # Annualize the volatility
        df_input.loc[df_input['Ticker'] == ticker_symbol, 'Annualized_Volatility'] = np.round(volatility, 2)

        # Calculate Sharpe Ratio
        risk_free_rate = 0.01  # Assuming a risk-free rate of 1%
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility
        df_input.loc[df_input['Ticker'] == ticker_symbol, 'Sharpe_Ratio'] = np.round(sharpe_ratio, 2)

        # Calculate Beta
        benchmark_ticker = '^GSPC'  # S&P 500 as benchmark
        benchmark = yf.Ticker(benchmark_ticker)
        benchmark_hist = benchmark.history(period=period).reset_index()
        benchmark_hist['Date'] = pd.to_datetime(benchmark_hist['Date'])
        benchmark_hist['Daily_Return'] = benchmark_hist['Close'].pct_change()
        hist = hist.merge(benchmark_hist[['Date', 'Daily_Return']], on='Date', suffixes=('', '_Benchmark'))
        covariance = hist['Daily_Return'].cov(hist['Daily_Return_Benchmark'])
        benchmark_variance = hist['Daily_Return_Benchmark'].var()
        beta = covariance / benchmark_variance if benchmark_variance != 0 else np.nan
        df_input.loc[df_input['Ticker'] == ticker_symbol, 'Beta'] = np.round(beta, 2)

        # Years since founded
        df_input.loc[df_input['Ticker'] == ticker_symbol, 'Years_Since_Founded'] = current_year - int(df_input[df_input['Ticker'] == ticker_symbol]['Founded'].max()[:4])

        return df_input
    
    except Exception as e:
        print(f"Error processing {ticker_symbol}: {e}")
        return df_input.sort_values('Market_Cap', ascending=False)

def save_table_to_bigquery(df, dataset_id, table_id):
    client = bigquery.Client()
    table_ref = client.dataset(dataset_id).table(table_id)
    
    # If the table does not exist, create it
    try:
        client.get_table(table_ref)
        print(f"Table {table_id} already exists in dataset {dataset_id}.")
    except NotFound:
        schema = create_schema(df)
        table = bigquery.Table(table_ref, schema=schema)
        table = client.create_table(table)
    print(f"Table {table_id} created in dataset {dataset_id}.")
    
    # Insert the DataFrame into the BigQuery table
    job = client.load_table_from_dataframe(df, table_ref)
    job.result()  # Wait for the job to complete

def load_table_from_bigquery(dataset_id, table_id, project_id):
    """Load a table from BigQuery."""

    query = f"SELECT * FROM `{dataset_id}.{table_id}`"
    df = pandas_gbq.read_gbq(query, project_id=project_id)
    return df