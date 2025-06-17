import streamlit as st
import pandas as pd
from typing import List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

from scripts.parameters import *
from scripts.functions.rag import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

# # --- Sidebar Configuration ---
# with st.sidebar:
#     st.header("⚙️ Configuration")
    
#     # API Key Configuration
#     st.subheader("API Keys")
#     with st.expander("Configure API Keys"):
#         st.info("Set your API keys in Streamlit secrets or environment variables:")
#         st.code("""
# # In .streamlit/secrets.toml
# PINECONE_API_KEY = "your_pinecone_key"
# GROQ_API_KEY = "your_groq_key"
#         """)
    
    # Model Configuration
    # st.subheader("Model Settings")
    
    # # Allow user to specify index name
    # pinecone_index_name = st.text_input(
    #     "Pinecone Index Name", 
    #     value="stock-portfolio-rag",
    #     help="Enter the name of your Pinecone index"
    # )
    
    # top_k = st.slider("Number of similar stocks to retrieve", 1, 10, 5)
    
    # # Check Pinecone connection
    # if st.button("🔍 Check Pinecone Connection"):
    #     try:
    #         from pinecone import Pinecone
    #         pinecone_api = get_api_key("PINECONE_API_KEY")
    #         if pinecone_api:
    #             pc = Pinecone(api_key=pinecone_api)
    #             available_indexes = pc.list_indexes()
    #             index_names = [idx.name for idx in available_indexes] if available_indexes else []
                
    #             if index_names:
    #                 st.success(f"✅ Found {len(index_names)} indexes:")
    #                 for name in index_names:
    #                     st.write(f"- {name}")
    #             else:
    #                 st.warning("⚠️ No indexes found in your Pinecone project")
    #         else:
    #             st.error("❌ Pinecone API key not found")
    #     except Exception as e:
    #         st.error(f"❌ Connection failed: {e}")
    
    # # System Status
    # st.subheader("System Status")
    # pinecone_key = get_api_key("PINECONE_API_KEY")
    # groq_key = get_api_key("GROQ_API_KEY")
    
    # pinecone_status = "🟢 Ready" if pinecone_key else "🔴 Missing API Key"
    # groq_status = "🟢 Ready" if groq_key else "🔴 Missing API Key"
    
    # st.write(f"**Pinecone:** {pinecone_status}")
    # st.write(f"**Groq LLM:** {groq_status}")
    
    # # Show environment info for debugging
    # if st.checkbox("Show Debug Info"):
    #     st.write("**Environment Variables:**")
    #     st.write(f"- PINECONE_API_KEY: {'✅ Set' if os.getenv('PINECONE_API_KEY') else '❌ Not set'}")
    #     st.write(f"- GROQ_API_KEY: {'✅ Set' if os.getenv('GROQ_API_KEY') else '❌ Not set'}")
        
    #     st.write("**Streamlit Secrets:**")
    #     try:
    #         pinecone_secret = st.secrets.get("PINECONE_API_KEY", "")
    #         groq_secret = st.secrets.get("GROQ_API_KEY", "")
    #         st.write(f"- PINECONE_API_KEY: {'✅ Set' if pinecone_secret else '❌ Not set'}")
    #         st.write(f"- GROQ_API_KEY: {'✅ Set' if groq_secret else '❌ Not set'}")
    #     except Exception as e:
    #         st.write(f"- Secrets not accessible: {e}")

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
            # Get the pinecone index name from sidebar
            index_name = pinecone_index_name
            
            # Perform RAG query
            response = rag_query_stocks(query=user_query, top_k=1, groq_llm_model=groq_llm_model,
                                        huggingface_embeddings_model=huggingface_embeddings_model,
                                        pinecone_index_name=index_name)
            
            if response['success']:
                # Display main answer
                st.success("✅ Analysis Complete!")
                
                # Main answer in prominent container
                with st.container():
                    st.subheader("🎯 Investment Insights")
                    st.markdown(response['answer'])
                
                # # Metrics
                # col1, col2, col3 = st.columns(3)
                # with col1:
                #     st.metric("Sources Found", response['num_sources'])
                # with col2:
                #     st.metric("Query Type", "RAG Analysis")
                # with col3:
                #     st.metric("Status", "Success")
                
                # Show context if enabled
                show_context = st.checkbox("Show retrieved context", value=True)
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