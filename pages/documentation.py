import streamlit as st

st.markdown("<h1 style='text-align: center; color: black;'>DOCUMENTATION</h1>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    #### 🤗 HuggingFace Embeddings

    **Purpose:** Text-to-vector conversion for semantic search
    
    **Resources:**
    - [HuggingFace Hub](https://huggingface.co/)
    - [Sentence Transformers](https://huggingface.co/sentence-transformers)
    - [Embeddings Guide](https://huggingface.co/blog/getting-started-with-embeddings)
    """)

with col2:
    st.markdown("""
    #### 🌲 Pinecone Vector Database
    
    **Purpose:** High-performance vector similarity search
                
    **Resources:**
    - [Pinecone Documentation](https://docs.pinecone.io/)
    - [Getting Started Guide](https://docs.pinecone.io/guides/getting-started/quickstart)
    - [Python SDK](https://docs.pinecone.io/guides/data/understanding-multitenancy)
    """)

with col3:
    st.markdown("""
    #### ⚡ LangChain ChatGroq
    
    **Purpose:** Fast language model inference
    
    **Resources:**
    - [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
    - [ChatGroq Integration](https://python.langchain.com/docs/integrations/chat/groq)
    - [Groq API Docs](https://console.groq.com/docs/quickstart)
    """)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    #### 🐙 GitHub Actions
    
    **Purpose:** CI/CD automation platform
    
    **Resources:**
    - [GitHub Actions Docs](https://docs.github.com/en/actions)
    - [Workflow Syntax](https://docs.github.com/en/actions/using-workflows)
    - [GCP Integration](https://github.com/google-github-actions)
    """)

with col2:
    st.markdown("""
    #### 🚀 Cloud Run
    
    **Purpose:** Serverless container platform
    
    **Resources:**
    - [Cloud Run Documentation](https://cloud.google.com/run/docs)
    - [Quickstart Guide](https://cloud.google.com/run/docs/quickstarts)
    - [Deployment Guide](https://cloud.google.com/run/docs/deploying)
    """)

with col3:
    st.markdown("""
    #### 📦 Artifact Registry
    
    **Purpose:** Container and package management
    
    **Resources:**
    - [Artifact Registry Docs](https://cloud.google.com/artifact-registry/docs)
    - [Docker Guide](https://cloud.google.com/artifact-registry/docs/docker)
    - [Security Features](https://cloud.google.com/artifact-registry/docs/analysis)
    """)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    #### 🗄️ BigQuery
    
    **Purpose:** Data warehouse and analytics platform
    
    **Resources:**
    - [BigQuery Documentation](https://cloud.google.com/bigquery/docs)
    - [Getting Started](https://cloud.google.com/bigquery/docs/quickstarts)
    - [SQL Reference](https://cloud.google.com/bigquery/docs/reference/standard-sql)
    """)
    
with col2:
    st.markdown("""
    #### 🐳 Docker
    
    **Purpose:** Containerization platform
    
    **Resources:**
    - [Docker Documentation](https://docs.docker.com/)
    - [Dockerfile Reference](https://docs.docker.com/engine/reference/builder/)
    - [Best Practices](https://docs.docker.com/develop/dev-best-practices/)
    """)

with col3:
    st.markdown("""
    #### ⚡ Prefect
    
    **Purpose:** Modern workflow orchestration
    
    **Resources:**
    - [Prefect Documentation](https://docs.prefect.io/)
    - [Getting Started](https://docs.prefect.io/latest/getting-started/)
    - [Cloud Platform](https://docs.prefect.io/latest/cloud/)
    """)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    #### 📊 Streamlit
    
    **Purpose:** Web application framework for data science
    
    **Resources:**
    - [Streamlit Documentation](https://docs.streamlit.io/)
    - [API Reference](https://docs.streamlit.io/library/api-reference)
    - [Gallery & Examples](https://streamlit.io/gallery)
    - [Deployment Guide](https://docs.streamlit.io/streamlit-community-cloud)
    """)

with col2:
    st.markdown("""
    #### 📈 Yahoo Finance
    
    **Purpose:** Financial data provider
    
    **Resources:**
    - [yfinance Library](https://pypi.org/project/yfinance/)
    - [Yahoo Finance](https://finance.yahoo.com/)
    - [GitHub Repository](https://github.com/ranaroussi/yfinance)
    - [Usage Examples](https://github.com/ranaroussi/yfinance#quick-start)
    """)

st.divider()

st.markdown("""
<div style='text-align: center; color: #666;'>
<p>📚 Complete documentation for the RAG-LLM Stock Portfolio Recommender System</p>
</div>
""", unsafe_allow_html=True)