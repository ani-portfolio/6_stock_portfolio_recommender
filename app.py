import streamlit as st
import pandas as pd
from typing import List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

from scripts.parameters import *
from scripts.functions.rag import *

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="Stock Portfolio RAG Assistant", 
    page_icon="📈",
    layout="wide"
)

# --- Header ---
st.title("📈 Stock Portfolio RAG Assistant")
st.markdown("""
**AI-Powered Stock Analysis & Portfolio Recommendations**  
Ask natural language questions about stocks and get insights from our comprehensive financial database.
""")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key Configuration
    st.subheader("API Keys")
    with st.expander("Configure API Keys"):
        st.info("Set your API keys in Streamlit secrets or environment variables:")
        st.code("""
# In .streamlit/secrets.toml
PINECONE_API_KEY = "your_pinecone_key"
GROQ_API_KEY = "your_groq_key"
        """)
    
    # Model Configuration
    st.subheader("Model Settings")
    top_k = st.slider("Number of similar stocks to retrieve", 1, 10, 5)
    show_context = st.checkbox("Show retrieved context", value=True)
    
    # System Status
    st.subheader("System Status")
    pinecone_status = "🟢 Ready" if st.secrets.get("PINECONE_API_KEY") else "🔴 Missing API Key"
    groq_status = "🟢 Ready" if st.secrets.get("GROQ_API_KEY") else "🔴 Missing API Key"
    
    st.write(f"**Pinecone:** {pinecone_status}")
    st.write(f"**Groq LLM:** {groq_status}")

# --- Session State Management ---
if 'selected_query' not in st.session_state:
    st.session_state.selected_query = ""

# --- Main Interface ---
col1, col2 = st.columns([3, 1])

with col1:
    # Use selected_query as default value if available
    default_value = st.session_state.selected_query if st.session_state.selected_query else ""
    
    # Query Input
    user_query = st.text_input(
        "🔍 Ask about stocks, sectors, or portfolio strategies:",
        placeholder="e.g., 'Which tech stocks have high returns and low risk?'",
        value=default_value,
        key="user_input"
    )
    
    # Clear the selected query after it's been used
    if st.session_state.selected_query:
        st.session_state.selected_query = ""

with col2:
    st.write("")  # Spacing
    search_button = st.button("🚀 Analyze", type="primary", use_container_width=True)

# --- Sample Queries ---
create_sample_queries()

# --- Process Query ---
if user_query and (search_button or user_query):
    with st.spinner("🔍 Analyzing stock data..."):
        try:
            # Perform RAG query
            response = rag_query_stocks(user_query, top_k, groq_llm_model, huggingface_embeddings_model, pinecone_index_name)
            
            if response['success']:
                # Display main answer
                st.success("✅ Analysis Complete!")
                
                # Main answer in prominent container
                with st.container():
                    st.subheader("🎯 Investment Insights")
                    st.markdown(response['answer'])
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sources Found", response['num_sources'])
                with col2:
                    st.metric("Query Type", "RAG Analysis")
                with col3:
                    st.metric("Status", "Success")
                
                # Show context if enabled
                if show_context and response['context']:
                    st.divider()
                    display_stock_cards(response['context'])
                
            else:
                st.error("❌ Analysis Failed")
                st.error(response['answer'])
                
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.info("Please check your API keys and ensure all dependencies are installed.")

# --- Footer ---
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
<p>Powered by RAG (Retrieval-Augmented Generation) • Pinecone Vector DB • Groq LLM • HuggingFace Embeddings</p>
</div>
""", unsafe_allow_html=True)