import os

os.chdir('..')
print(os.getcwd())
# Import necessary libraries
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import timedelta
import datetime as dt
from airflow.utils.dates import days_ago
import fnmatch
import yfinance as yf
from google.cloud import storage

from src.scripts import update_stock_data

default_args = {
    'owner': 'Ani',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'start_date':  days_ago(1),
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG('Stock_data',
         start_date=days_ago(1),
         description='A DAG to fetch stock data from Yahoo Finance and store it in BigQuery',
         # Run once at 6pm EST
         schedule_interval= '0 21 * * *',  # 9PM UTC (5PM EST)
         catchup=False, 
         default_args=default_args, 
         tags=["bq"]
) as dag:

    # Get base data from the URL
    create_stock_dataset = PythonOperator(
        task_id = 'fetch_stock_data',
        python_callable = update_stock_data)
    
    (
        create_stock_dataset
    )