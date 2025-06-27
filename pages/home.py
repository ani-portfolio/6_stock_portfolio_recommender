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
from src.rag import *


# --- Custom CSS for orange input box ---
st.markdown("""
<style>
/* Style the text input box */
.stTextInput > div > div > input {
    background-color: white !important;
    border: 2px solid #ddd !important;
    border-radius: 8px !important;
    color: #2d3436 !important;
    font-weight: 500 !important;
}

/* Input box focus state */
.stTextInput > div > div > input:focus {
    background-color: white !important;
    border-color: #007bff !important;
    box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25) !important;
}

/* Placeholder text styling */
.stTextInput > div > div > input::placeholder {
    color: #636e72 !important;
    opacity: 0.8 !important;
}

/* Input label styling */
.stTextInput > label {
    color: #666 !important;
    font-weight: 600 !important;
}

/* Button height matching input box */
.stButton > button {
    height: 48px !important;
    margin-top: 0px !important;
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
# st.title("STOCK RECOMMENDATION SYSTEM")
st.markdown("<h1 style='text-align: center; color: black;'>STOCK RECOMMENDATION SYSTEM</h1>", unsafe_allow_html=True)

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
            response = rag_query_stocks(query=user_query, top_k=10, groq_llm_model=groq_llm_model,
                                        huggingface_embeddings_model=huggingface_embeddings_model,
                                        pinecone_index_name=index_name)
            
            if response['success']:
                # Display main answer
                st.success("Analysis Complete")
                
                # Main answer in prominent container
                with st.container():
                    # st.subheader("🎯 Investment Insights")
                    st.markdown(response['answer'])

                    st.markdown(
                            """
                            <style>
                            .disclaimer-box {
                                background-color: #ffebee;
                                border: 2px solid #f44336;
                                border-radius: 8px;
                                padding: 16px;
                                margin-top: 24px;
                                text-align: center;
                                box-shadow: 0 2px 4px rgba(244, 67, 54, 0.1);
                            }
                            .disclaimer-text {
                                color: #d32f2f;
                                font-weight: 600;
                                margin: 0;
                                text-align: center;
                                font-size: 15px;
                            }
                            </style>
                            <div class="disclaimer-text">
                                Please note that this information is not financial advice and should not be considered as such
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                
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