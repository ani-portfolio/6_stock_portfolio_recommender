import streamlit as st
import os
import base64

# Set environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="Stock Recommendation System", 
    page_icon="📈",
    layout="centered"
)

# --- Background Image ---
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    
    /* Extend background to navigation bar */
    .stAppHeader, 
    header[data-testid="stHeader"],
    .stAppHeader > div,
    section[data-testid="stSidebar"] > div:first-child {{
        background: transparent !important;
    }}
    
    /* Make navigation bar transparent */
    .stTabs [data-baseweb="tab-list"] {{
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 10px !important;
        padding: 5px !important;
        margin-bottom: 20px !important;
    }}
    
    /* Style navigation tabs */
    .stTabs [data-baseweb="tab"] {{
        background: transparent !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        margin: 0 2px !important;
    }}
    
    /* Active tab styling */
    .stTabs [aria-selected="true"] {{
        background: rgba(255, 255, 255, 0.2) !important;
        color: white !important;
    }}
    
    /* Hover effect for tabs */
    .stTabs [data-baseweb="tab"]:hover {{
        background: rgba(255, 255, 255, 0.15) !important;
    }}
    
    /* Remove default Streamlit header background */
    .css-1rs6os, .css-17ziqus {{
        background: transparent !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

# Apply background (make sure you have background.jpg/png in your project folder)
try:
    add_bg_from_local('media/background.jpg')  # Change to your image file
except:
    st.info("Background image not found - using default background")

# --- Navigation ---
home_page = st.Page("pages/home.py", title="Home")
summary_page = st.Page("pages/summary.py", title="Risk Level")
documentation_page = st.Page("pages/documentation.py", title="Documentation")
links_page = st.Page("pages/links.py", title="Links")
architecture_page = st.Page("pages/architecture.py", title="System Architecture")

pg = st.navigation([home_page, summary_page, architecture_page, documentation_page, links_page], position="top")
pg.run()
