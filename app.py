import streamlit as st
import os
import base64

os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(
    page_title="Stock Recommendation System", 
    page_icon="📈",
    layout="centered"
)

try:
    with open('media/background.jpg', "rb") as image_file:
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
    
    .stAppHeader, 
    header[data-testid="stHeader"],
    .stAppHeader > div,
    section[data-testid="stSidebar"] > div:first-child {{
        background: transparent !important;
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 10px !important;
        padding: 5px !important;
        margin-bottom: 20px !important;
        display: flex !important;
        justify-content: center !important;
        width: 100% !important;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: transparent !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        margin: 0 2px !important;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: rgba(255, 255, 255, 0.2) !important;
        color: white !important;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background: rgba(255, 255, 255, 0.15) !important;
    }}
    
    .css-1rs6os, .css-17ziqus {{
        background: transparent !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
except:
    st.info("Background image not found - using default background")

home_page = st.Page("pages/home.py", title="Home")
summary_page = st.Page("pages/summary.py", title="Risk Level")
documentation_page = st.Page("pages/documentation.py", title="Documentation")
links_page = st.Page("pages/links.py", title="Links")
architecture_page = st.Page("pages/architecture.py", title="System Architecture")

pg = st.navigation([home_page, summary_page, architecture_page, documentation_page, links_page], position="top")
pg.run()
