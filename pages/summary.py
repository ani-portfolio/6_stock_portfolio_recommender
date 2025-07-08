import streamlit as st
import pandas as pd
import sys

sys.path.append('..')
from src.parameters import *
from src.data_ingestion import *
from src.charts import *
from src.utils import *

st.set_page_config(layout="wide")

st.markdown("""<h1 style='text-align: center; color: black;'>RISK LEVEL</h1>""", unsafe_allow_html=True)

df_stock_data = load_table_from_bigquery(dataset_id, table_id, project_id)
df_top_25_summary = create_summary_chart(df_stock_data)

if df_top_25_summary is not None and not df_top_25_summary.empty:
    
    df_clean = df_top_25_summary.copy()
    df_clean.columns = df_clean.columns.str.replace('_', ' ')
    
    styled_df = df_clean.style
    
    styled_df = styled_df.format("{:.1f}", subset=pd.IndexSlice[:, df_clean.select_dtypes(include=[float]).columns])
    
    if 'Risk' in df_clean.columns:
        styled_df = styled_df.applymap(highlight_risk, subset=['Risk'])
    
    if 'Sentiment' in df_clean.columns:
        styled_df = styled_df.applymap(highlight_sentiment, subset=['Sentiment'])
    
    styled_df = styled_df.set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center')]},
        {'selector': 'th.col_heading', 'props': [('text-align', 'center')]}
    ]).hide(axis="index")
    
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

st.markdown("""
<div style='text-align: center; color: #666;'>
<p> Top 25 Stocks by Market Cap </p>
</div>
""", unsafe_allow_html=True)