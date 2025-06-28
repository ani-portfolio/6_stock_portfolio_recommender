import streamlit as st
import pandas as pd
from typing import List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

def get_api_key(key_name: str) -> str:
    """
    Get API key from environment variables or Streamlit secrets
    Prioritizes environment variables for Cloud Run compatibility
    """
    # First try environment variables (works in Cloud Run)
    env_value = os.getenv(key_name)
    if env_value:
        return env_value
    
    # Fallback to Streamlit secrets (local development)
    try:
        return st.secrets.get(key_name, "")
    except Exception:
        # If st.secrets fails (e.g., no secrets.toml), return empty string
        return ""

def initialize_rag_components(pinecone_api: str, 
                              groq_api: str, 
                              groq_llm_model: str, 
                              huggingface_embeddings_model: str,
                              pinecone_index_name: str) -> tuple:
    """Initialize all RAG components: Pinecone, Groq LLM, and HuggingFace embeddings"""
    try:
        from pinecone import Pinecone
        from langchain_groq import ChatGroq
        from langchain_huggingface import HuggingFaceEmbeddings
        
        # Initialize Pinecone
        pc = Pinecone(api_key=pinecone_api)
        index = pc.Index(pinecone_index_name)

        # Initialize Groq LLM
        llm = ChatGroq(
            groq_api_key=groq_api,
            model=groq_llm_model,
            temperature=0.1,
            max_tokens=1000
        )
        
        # Initialize HuggingFace embeddings (same as used in indexing)
        embeddings = HuggingFaceEmbeddings(
            model_name=huggingface_embeddings_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        return index, llm, embeddings
    except ImportError as e:
        st.error(f"Missing required packages: {e}")
        return None, None, None

def retrieve_relevant_context(query: str, index, embeddings, top_k: int = 5) -> List[Dict]:
    """Retrieve relevant stock data from Pinecone"""
    
    try:
        # Convert query to embedding
        query_embedding = embeddings.embed_query(query)
        
        # Search Pinecone for similar vectors
        search_results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            include_values=False
        )
        
        # Extract relevant documents
        retrieved_docs = []
        for match in search_results.matches:
            retrieved_docs.append({
                'score': match.score,
                'content': match.metadata.get('content', 'Content not available'),
                'metadata': {
                    'Ticker': match.metadata.get('Ticker'),
                    'Company_Name': match.metadata.get('Company_Name'),
                    'Sector': match.metadata.get('Sector'),
                    'Industry': match.metadata.get('Industry'),
                },
                'id': match.id
            })
        
        return retrieved_docs
        
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return []

def format_context_for_llm(retrieved_docs: List[Dict]) -> str:
    """Format retrieved documents for LLM context"""
    
    if not retrieved_docs:
        return "No relevant stock data found."
    
    context_parts = []
    
    for i, doc in enumerate(retrieved_docs, 1):
        context_parts.append(f"=== Stock Data {i} (Relevance: {doc['score']:.3f}) ===")
        context_parts.append(doc['content'])  # This contains all the formatted stock info
        context_parts.append("")  # Spacing
    
    return "\n".join(context_parts)

def create_stock_prompt_template():
    """Create a prompt template for stock-related queries"""
    from langchain.prompts import PromptTemplate
    
    template = """
You are a knowledgeable financial advisor assistant. Based on the stock data provided below, answer the user's question accurately and concisely.

CONTEXT (Retrieved Stock Data):
{context}

USER QUESTION: {question}

INSTRUCTIONS:
1. Answer based ONLY on the provided stock data
2. If the information isn't available in the context, clearly state that
3. Provide specific numbers when available (percentages, dollar amounts, etc.)
4. If multiple stocks are relevant, compare them
5. Be conversational but professional
6. Present the data in a well-formatted table
7. Use markdown table format with | separators
8. Include relevant column headers
9. Show numerical data with appropriate precision
10. If comparing multiple items, each should be a separate row
11. Include units (%, $, etc.) in column headers when applicable
12. Always return Company_Name
13. Replace '_' with ' ' in column headers for readability
14. Do not mention anything about the context or how the data was retrieved
15. Answer the query in text format first, then provide the table
16. At the end of the answer, add the following disclaimer "Please note that this is based on 10 years of historical data last updated on Update_Date"

Use the following table format for your answer. Always include the 4 fields Ticker, Closing_Price, All_Time_High, Percent_From_All_Time_High but feel free to add any other additional pertinent fields:
| Ticker | Closing_Price | All_Time_High  | Percent_From_All_Time_High
|---------|-------------|------------|-----------|
| AAPL    | $150.25     | $28.15      | 10%     


ANSWER:
"""
    
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

def rag_query_stocks(query, top_k, groq_llm_model, huggingface_embeddings_model, pinecone_index_name):
    """Enhanced RAG function using stored page content"""
    
    try:
        print(f"Query: {query}")
        
        # Get API keys from environment or Streamlit secrets
        pinecone_api = get_api_key("PINECONE_API_KEY")
        groq_api = get_api_key("GROQ_API_KEY")

        
        # Initialize components
        index, llm, embeddings = initialize_rag_components(
            pinecone_api=pinecone_api,
            groq_api=groq_api,
            groq_llm_model=groq_llm_model,
            huggingface_embeddings_model=huggingface_embeddings_model,
            pinecone_index_name=pinecone_index_name
        )
        
        if not all([index, llm, embeddings]):
            return {
                "answer": "Failed to initialize RAG components. Please check your API keys and configuration.",
                "context": [],
                "query": query,
                "success": False
            }
        
        # Use enhanced retrieval with stored content
        print("Retrieving relevant stock data with full content...")
        retrieved_docs = retrieve_relevant_context(query, index, embeddings, top_k)
        
        if not retrieved_docs:
            return {
                "answer": "I couldn't find any relevant stock data for your query. Please check if the stock is in our database.",
                "context": [],
                "query": query,
                "success": False
            }
        
        # Format context using full content
        context_text = format_context_for_llm(retrieved_docs)
        
        # Create prompt
        prompt_template = create_stock_prompt_template()
        formatted_prompt = prompt_template.format(
            context=context_text,
            question=query
        )
        
        # Generate answer using Groq
        print("Generating answer with Groq...")
        response = llm.invoke(formatted_prompt)
        answer = response.content
        
        return {
            "answer": answer,
            "context": retrieved_docs,
            "query": query,
            "num_sources": len(retrieved_docs),
            "success": True
        }
        
    except Exception as e:
        return {
            "answer": f"An error occurred while processing your query: {str(e)}",
            "context": [],
            "query": query,
            "success": False
        }

def display_stock_cards(retrieved_docs: List[Dict]):
    """Display retrieved stock information as cards"""
    
    if not retrieved_docs:
        return
    
    st.subheader("📊 Retrieved Stock Information")
    
    for i, doc in enumerate(retrieved_docs):
        metadata = doc['metadata']
        
        with st.expander(f"{metadata.get('Company_Name', 'Unknown')} ({metadata.get('Ticker', 'N/A')})"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Sector:** {metadata.get('Sector', 'N/A')}")
                st.write(f"**Industry:** {metadata.get('Industry', 'N/A')}")

            with col2:
                st.write(f"**Relevance Score:** {doc['score']:.3f}")
            
            st.write("**Full Content:**")
            st.text(doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content'])

def create_sample_queries():
    """Create sample query buttons for users"""
    
    sample_queries = [
        "What is NVIDIA's annualized return?",
        "What is Apple's return in 2024?",
        "Which stocks have the highest annualized returns in 2024?",
        "Show me dividend-paying stocks with low volatility",
        "Find stocks with a P/E ratio below 20 and annualized return above 10%",
        "Find undervalued stocks trading below their 52-week high"
    ]
    
    st.markdown("""**Sample Queries** """)
    cols = st.columns(2)
    for i, query in enumerate(sample_queries):
        col = cols[i % 2]
        if col.button(query, key=f"sample_{i}", use_container_width=True):
            st.session_state.selected_query = query
            st.rerun()