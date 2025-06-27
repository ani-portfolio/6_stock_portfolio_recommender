import streamlit as st
import pandas as pd
from typing import List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Import your modules (adjust the import path as needed)
import sys
sys.path.append('..')
from src.parameters import *
from src.data_ingestion import *

st.set_page_config(
    # page_title="Stock Recommendation System", 
    # page_icon="📈",
    layout="wide"
)

# Function to highlight Risk column values
def highlight_risk(val):
    """
    Highlight Risk column values: Red for High, Yellow for Medium, Green for Low.
    """
    if val == 'High':
        return 'background-color: #FFB6C1'  # Light red
    elif val == 'Medium':
        return 'background-color: #FFFFE0'  # Light yellow
    elif val == 'Low':
        return 'background-color: #90EE90'  # Light green
    return ''

# --- Summary Page ---
st.markdown("""<h1 style='text-align: center; color: black;'>RISK LEVEL</h1>""", unsafe_allow_html=True)

# Load data and create summary
df_stock_data = load_table_from_bigquery(dataset_id, table_id, project_id)
df_top_25_summary = create_summary_chart(df_stock_data)

# Display the dataframe as a table
if df_top_25_summary is not None and not df_top_25_summary.empty:
    
    # Clean column names by removing underscores and replacing with spaces
    df_clean = df_top_25_summary.copy()
    df_clean.columns = df_clean.columns.str.replace('_', ' ')
    
    # Apply styling
    styled_df = df_clean.style
    
    # Format numeric columns to 1 decimal place
    styled_df = styled_df.format("{:.1f}", subset=pd.IndexSlice[:, df_clean.select_dtypes(include=[float]).columns])
    
    # Highlight Risk column if it exists
    if 'Risk' in df_clean.columns:
        styled_df = styled_df.applymap(highlight_risk, subset=['Risk'])
    
    # Center all column headers and hide index
    styled_df = styled_df.set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center')]},
        {'selector': 'th.col_heading', 'props': [('text-align', 'center')]}
    ]).hide(axis="index")
    
    # Display using st.dataframe for interactive table with styling
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=600,
        hide_index=True
    )

else:
    st.error("❌ No data available to display")
    st.info("Please check your data source and try again.")

st.divider()

# --- Footer ---
st.markdown("""
<div style='text-align: center; color: #666;'>
<p> Top 25 Stocks by Market Cap </p>
</div>
""", unsafe_allow_html=True)