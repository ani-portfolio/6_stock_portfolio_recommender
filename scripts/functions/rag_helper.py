
import pandas as pd

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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

def setup_rag_system(documents, embedding_model, llm_model, persist_directory):
    """
    Set up the RAG system with ChromaDB and SentenceTransformer embeddings.
    """
    # Initialize embeddings
    embedding_function = SentenceTransformerEmbeddings(model_name=embedding_model)
    
    # Create LangChain documents
    langchain_documents = [
        Document(page_content=doc["page_content"], metadata=doc["metadata"])
        for doc in documents
    ]

    # Initialize ChromaDB
    print(f"\nInitializing ChromaDB at: {persist_directory}")

    try:
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
        if vectorstore._collection.count() != 0: 
            print("ChromaDB already exists. Checking for existing documents...")
            vectorstore._collection.delete()  # Clear existing collection
            print("Existing ChromaDB collection cleared.")
        else:
            print("ChromaDB is empty. Adding documents...")
            vectorstore.add_documents(langchain_documents)
            print(f"Added {len(langchain_documents)} documents to ChromaDB.")
    except Exception as e:
        print(f"Error loading ChromaDB, attempting to create new: {e}")
        vectorstore = Chroma.from_documents(
            langchain_documents,
            embedding_function,
            persist_directory=persist_directory
        )
        print(f"Created new ChromaDB and added {len(langchain_documents)} documents.")


    print("\nVector database (ChromaDB) setup complete.")

    llm = Ollama(model=llm_model)

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