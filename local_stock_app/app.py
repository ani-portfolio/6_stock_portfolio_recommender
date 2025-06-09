import streamlit as st

import pandas as pd

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import os # For checking if chroma_db directory exists

# --- Configuration ---
PERSIST_DIRECTORY = "/Users/ani/Projects/6_stock_portfolio_recommendation/chroma_db" # Directory to store ChromaDB
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Sentence Transformers model
LLM_MODEL = "mistral" # Ensure this model is pulled in Ollama (ollama pull mistral)
PATH = '/Users/ani/Projects/6_stock_portfolio_recommendation/data/sp500_data_sample.csv'

@st.cache_resource
def load_stock_data(path):
    """
    Load stock data from a CSV file.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={'Security': 'Company_Name'})

    return df

@st.cache_resource
def create_text_chunks(df):
    """
    Create text chunks from the stock data DataFrame.
    """

    documents = []
    columns = df.columns.tolist()

    for index, row in df.iterrows():

        doc_content = ""
        for col in columns:
            if pd.notna(row[col]):
                doc_content += f"{col}: {row[col]}\n"
        
        documents.append({
            "page_content": doc_content,
            "metadata": {"Ticker": row['Ticker'], "Company_Name": row['Company_Name']}
        })

    print("\nExample of a processed document for RAG:")
    print(documents[0]['page_content'])
    print(documents[0]['metadata'])
    
    return documents

@st.cache_resource
def setup_rag_system(documents):
    """
    Set up the RAG system with ChromaDB and SentenceTransformer embeddings.
    """
    # Initialize embeddings
    embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Create LangChain documents
    langchain_documents = [
        Document(page_content=doc["page_content"], metadata=doc["metadata"])
        for doc in documents
    ]

    # Initialize ChromaDB
    persist_directory = "/Users/ani/Projects/6_stock_portfolio_recommendation/chroma_db"

    print(f"\nInitializing ChromaDB at: {persist_directory}")

    try:
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
        if vectorstore._collection.count() == 0: 
            print("ChromaDB is empty. Adding documents...")
            vectorstore.add_documents(langchain_documents)
            print(f"Added {len(langchain_documents)} documents to ChromaDB.")
        else:
            print(f"ChromaDB already contains {vectorstore._collection.count()} documents. Skipping addition.")
    except Exception as e:
        print(f"Error loading ChromaDB, attempting to create new: {e}")
        vectorstore = Chroma.from_documents(
            langchain_documents,
            embedding_function,
            persist_directory=persist_directory
        )
        print(f"Created new ChromaDB and added {len(langchain_documents)} documents.")


    print("\nVector database (ChromaDB) setup complete.")

    llm = Ollama(model="mistral")

    print(f"\nOllama LLM initialized with model: {llm.model}")

    # Retriever setup
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Define the prompt template for the LLM
    prompt_template = ChatPromptTemplate.from_template("""
    Answer the question based ONLY on the following context.
    If the answer cannot be found in the context, politely state that you don't have enough information.

    Context:
    {context}

    Question:
    {question}
    """)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    print("\nRAG chain built successfully.")

    return rag_chain

# --- Streamlit App UI ---
st.set_page_config(page_title="Local Stock RAG App", layout="centered")

st.title("📈 Local Stock Insight Assistant")
st.markdown("Ask natural language questions about our sample stock data!")

# Load data and set up RAG system once
data = load_stock_data(PATH)
documents = create_text_chunks(data)
rag_chain = setup_rag_system(documents)

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