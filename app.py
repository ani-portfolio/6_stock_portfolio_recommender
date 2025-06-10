import streamlit as st

from scripts.functions.data_ingestion import *
from scripts.parameters import *
from scripts.functions.rag_helper import *

# --- Streamlit App UI ---
st.set_page_config(page_title="Local Stock RAG App", layout="centered")

st.title("📈 Local Stock Insight Assistant")
st.markdown("Ask natural language questions about our sample stock data!")

# Load data and set up RAG system once
data = load_table_from_bigquery(dataset_id=dataset_id, table_id=table_id, project_id=PROJECT_ID)
documents = create_text_chunks(data)
rag_chain = setup_rag_system(documents, EMBEDDING_MODEL, LLM_MODEL, PERSIST_DIRECTORY)

st.divider()

user_query = st.text_input("Enter your question about stocks (e.g., 'Which tech stocks have high annualized returns?', 'Tell me about NVIDIA.')", key="user_input")

if user_query:
    with st.spinner("Getting insights..."):
        try:
            response = rag_chain.invoke(user_query)
            st.success("Here's what I found:")
            st.info(response)
        except Exception as e:
            st.error(f"An error occurred during RAG query: {e}")
            st.warning("Please ensure Ollama is running and the specified model is downloaded.")

st.divider()
st.caption("Powered by Ollama, LangChain, ChromaDB, and Streamlit.")