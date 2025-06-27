from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_gbq
from typing import Dict, Optional

def get_tickers(url):
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
    return df['Ticker'].unique().tolist()

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

def _calculate_price_metrics(hist: pd.DataFrame) -> Dict:
    """Calculate price-related metrics."""
    closing_price = hist['Close'].iloc[-1]
    all_time_high = hist['Close'].max()
    
    # Calculate moving average
    hist['200_Day_Moving_Average'] = hist['Close'].rolling(window=200).mean()
    
    return {
        'Closing_Price': round(closing_price, 2),
        'All_Time_High': round(all_time_high, 2),
        'Percent_From_All_Time_High': round(((closing_price - all_time_high) / all_time_high) * 100, 2),
        'Percent_Difference_200_Day_Moving_Average': round(((closing_price - hist['200_Day_Moving_Average'].iloc[-1]) / hist['200_Day_Moving_Average'].iloc[-1]) * 100, 2),
        '24_Hour_Percent_Change': round(hist['Close'].pct_change(periods=1).iloc[-1] * 100, 2),
        '7_Day_Percent_Change': round(hist['Close'].pct_change(periods=7).iloc[-1] * 100, 2),
        '30_Day_Percent_Change': round(hist['Close'].pct_change(periods=30).iloc[-1] * 100, 2)
    }


def _calculate_returns(hist: pd.DataFrame, current_year: int) -> Dict:
    """Calculate return metrics."""
    returns_data = {}
    
    # Annualized return
    total_years = len(hist['Year'].unique())
    if total_years > 1:
        annualized_return = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) ** (1 / total_years) - 1) * 100
        returns_data['Annualized_Return'] = round(annualized_return, 2)
    
    # Yearly returns
    for year in sorted(hist['Year'].unique(), reverse=True):
        year_data = hist[hist['Year'] == year]
        if len(year_data) > 1:
            year_return = ((year_data['Close'].iloc[-1] - year_data['Close'].iloc[0]) / year_data['Close'].iloc[0]) * 100
            
            if year == current_year:
                returns_data['YTD_Return'] = round(year_return, 2)
            else:
                returns_data[f'{year}_Return'] = round(year_return, 2)
    
    return returns_data


def _calculate_risk_metrics(hist: pd.DataFrame, benchmark_data: Optional[pd.DataFrame] = None) -> Dict:
    """Calculate risk-related metrics."""
    # Daily returns
    hist['Daily_Return'] = hist['Close'].pct_change().dropna()
    
    # Volatility (annualized)
    volatility = hist['Daily_Return'].std() * np.sqrt(252)
    
    # Sharpe ratio (assuming 1% risk-free rate)
    risk_free_rate = 0.01
    if len(hist) > 1:
        total_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(hist)) - 1
        excess_return = annualized_return - risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
    else:
        sharpe_ratio = 0
    
    # Beta calculation (use pre-loaded benchmark data if available)
    beta = _calculate_beta(hist, benchmark_data)
    
    return {
        'Annualized_Volatility': round(volatility * 100, 2),
        'Sharpe_Ratio': round(sharpe_ratio, 2),
        'Beta': round(beta, 2) if not np.isnan(beta) else None
    }


def _calculate_beta(hist: pd.DataFrame, benchmark_data: Optional[pd.DataFrame] = None) -> float:
    """Calculate beta against S&P 500 using pre-loaded benchmark data."""
    try:
        if benchmark_data is None:
            return np.nan
            
        # Merge on date
        merged = hist.merge(
            benchmark_data[['Date', 'Daily_Return']], 
            on='Date', 
            suffixes=('', '_Benchmark'),
            how='inner'
        )
        
        if len(merged) < 30:  # Need sufficient data points
            return np.nan
        
        # Calculate beta using numpy for speed
        stock_returns = merged['Daily_Return'].dropna()
        benchmark_returns = merged['Daily_Return_Benchmark'].dropna()
        
        if len(stock_returns) == len(benchmark_returns) and len(stock_returns) > 0:
            covariance = np.cov(stock_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            return covariance / benchmark_variance if benchmark_variance != 0 else np.nan
        
        return np.nan
        
    except Exception:
        return np.nan

def _get_market_data(info: Dict) -> Dict:
    """Extract market data from ticker info."""
    return {
        'Company_Name': info.get('shortName', '').replace('\n', ' '),
        'Market_Cap': info.get('marketCap'),
        'Sector': info.get('sector'),
        'Industry': info.get('industry'),
        'Country': info.get('country'),
        'Business_Summary': info.get('longBusinessSummary', '').replace('\n', ' '),
        'Dividend_Yield': info.get('dividendYield'),
        'Trailing_PE': info.get('trailingPE'),
        'Forward_PE': info.get('forwardPE'),
        'Average_Volume': info.get('averageVolume'),
        'Average_Volume_10days': info.get('averageVolume10days'),
        '52_Week_Change': info.get('52WeekChange')
    }

def get_stock_data(ticker_symbol: str, period: str = "5y", benchmark_data: Optional[pd.DataFrame] = None) -> Dict:
    """
    Calculate comprehensive stock metrics for a given ticker.
    
    Args:
        ticker_symbol (str): Stock ticker symbol (e.g., 'AAPL')
        period (str): Historical data period (default: '5y')
        benchmark_data (pd.DataFrame): Pre-loaded S&P 500 data for beta calculation
        
    Returns:
        Dict: Dictionary containing all calculated metrics for the ticker
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # Get historical data
        hist = ticker.history(period=period)
        if hist.empty:
            raise ValueError(f"No data found for {ticker_symbol}")
        
        hist = hist.reset_index()
        hist['Date'] = pd.to_datetime(hist['Date'])
        hist['Year'] = hist['Date'].dt.year
        
        # Get ticker info
        info = ticker.info
        current_year = hist['Year'].max()
        
        # Initialize results dictionary
        stock_data = {'Ticker': ticker_symbol}
        
        # Price-based metrics
        stock_data.update(_calculate_price_metrics(hist))
        
        # Return calculations
        stock_data.update(_calculate_returns(hist, current_year))
        
        # Risk metrics (pass benchmark_data to avoid reloading)
        stock_data.update(_calculate_risk_metrics(hist, benchmark_data))
        
        # Market data
        stock_data.update(_get_market_data(info))
        
        return stock_data
        
    except Exception as e:
        print(f"Error processing {ticker_symbol}: {e}")

def save_table_to_bigquery(df, dataset_id, table_id):
    client = bigquery.Client()
    table_ref = client.dataset(dataset_id).table(table_id)
    
    # If the table does not exist, create it
    try:
        client.get_table(table_ref)
        print(f"Table {table_id} already exists in dataset {dataset_id}.")
        # delete the existing table
        client.delete_table(table_ref)
        print(f"Table {table_id} deleted from dataset {dataset_id}.")
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

def create_summary_chart(df):
    df_result = df.copy()
    
    # Create the 'Risk' column based on the specified conditions
    def determine_Risk_signal(row):
        all_time_high_condition = row['Percent_From_All_Time_High'] < -10
        moving_avg_condition = row['Percent_Difference_200_Day_Moving_Average'] < 0
        
        # If Percent_From_All_Time_High is greater than -10%, then 'High'
        if row['Percent_From_All_Time_High'] >= -10:
            return 'High'
        
        # If both conditions are met (< -10% and negative moving avg), then 'Low'
        if all_time_high_condition and moving_avg_condition:
            return 'Low'
        
        # If only one condition is met, then 'Medium'
        if all_time_high_condition or moving_avg_condition:
            return 'Medium'
        
        # Default case (shouldn't reach here based on logic, but safety)
        return 'High'
    
    # Apply the Risk signal logic
    df_result['Risk'] = df_result.apply(determine_Risk_signal, axis=1)
    
    # Get top 25 by Market Cap first
    df_top_25 = df_result.sort_values('Market_Cap', ascending=False).head(25)
    
    # Create categorical ordering for Risk column
    Risk_order = ['Low', 'Medium', 'High']
    df_top_25['Risk'] = pd.Categorical(df_top_25['Risk'], categories=Risk_order, ordered=True)
    
    # Sort by Risk column (Low, Medium, High), then by Market Cap descending
    df_sorted = df_top_25.sort_values(['Risk', 'Market_Cap'], ascending=[True, False])
    
    # Return the sorted dataframe with selected columns
    return df_sorted[
        ['Ticker', 'Company_Name', 'Market_Cap', 'Closing_Price', 'All_Time_High', 'Risk',
         'Percent_From_All_Time_High', 'Percent_Difference_200_Day_Moving_Average', 
         '24_Hour_Percent_Change', '7_Day_Percent_Change', '30_Day_Percent_Change']
    ].reset_index(drop=True)