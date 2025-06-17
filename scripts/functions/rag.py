import streamlit as st
import pandas as pd
from typing import List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

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
                    'ticker': match.metadata.get('ticker'),
                    'company_name': match.metadata.get('company_name'),
                    'sector': match.metadata.get('sector'),
                    'industry': match.metadata.get('industry'),
                    'update_date': match.metadata.get('update_date')
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

ANSWER:
"""
    
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

def rag_query_stocks(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Enhanced RAG function using stored page content"""
    
    try:
        print(f"Query: {query}")
        
        # Get API keys from environment or Streamlit secrets
        pinecone_api = st.secrets.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY")
        groq_api = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        
        # Configuration parameters
        groq_llm_model = "mixtral-8x7b-32768"
        huggingface_embeddings_model = "sentence-transformers/all-MiniLM-L6-v2"
        pinecone_index_name = "stock-portfolio-rag"
        
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
        
        with st.expander(f"{metadata.get('company_name', 'Unknown')} ({metadata.get('ticker', 'N/A')}) - Relevance: {doc['score']:.3f}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Sector:** {metadata.get('sector', 'N/A')}")
                st.write(f"**Industry:** {metadata.get('industry', 'N/A')}")
            
            with col2:
                st.write(f"**Last Update:** {metadata.get('update_date', 'N/A')}")
                st.write(f"**Relevance Score:** {doc['score']:.3f}")
            
            st.write("**Full Content:**")
            st.text(doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content'])

def create_sample_queries():
    """Create sample query buttons for users"""
    
    sample_queries = [
        "What are the best performing tech stocks?",
        "Show me dividend-paying stocks with low volatility",
        "Which stocks have the highest annualized returns?",
        "Tell me about NVIDIA's performance",
        "Compare healthcare vs technology sector stocks",
        "Find undervalued stocks trading below their 52-week high"
    ]
    
    st.subheader("💡 Try These Sample Queries")
    
    cols = st.columns(2)
    for i, query in enumerate(sample_queries):
        col = cols[i % 2]
        if col.button(query, key=f"sample_{i}"):
            st.session_state.user_input = query
            st.rerun()