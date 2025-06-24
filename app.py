import streamlit as st
import os

# Set environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="Stock Recommendation System", 
    page_icon="📈",
    layout="centered"
)

# --- Navigation ---
home_page = st.Page("pages/home.py", title="Home")
documentation_page = st.Page("pages/documentation.py", title="Documentation")
links_page = st.Page("pages/links.py", title="Links")
architecture_page = st.Page("pages/architecture.py", title="System Architecture")

pg = st.navigation([home_page, architecture_page, documentation_page, links_page], position="top")
pg.run()
