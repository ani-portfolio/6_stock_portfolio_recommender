import streamlit as st
from PIL import Image
import os

import sys
sys.path.append('..')

# --- Architecture Page ---
# st.title("SYSTEM ARCHITECTURE")
st.markdown("<h1 style='text-align: center; color: black;'>SYSTEM ARCHITECTURE</h1>", unsafe_allow_html=True)

# st.markdown("""
# **BackEnd • FrontEnd**  
# """)
# st.markdown("---")

# Load and display architecture image
try:
    # Try to load the architecture image from media folder
    image_path = "media/architecture.png"
    if os.path.exists(image_path):
        architecture_image = Image.open(image_path)
        st.image(architecture_image, 
                use_container_width=True)
    else:
        st.warning("⚠️ Architecture image not found at: media/architecture.png")
        st.info("Please ensure the architecture.png file is placed in the media/ folder")
except Exception as e:
    st.error(f"❌ Error loading architecture image: {e}")

st.markdown("---")

st.markdown("""
<div style='text-align: center; color: #666;'>
<p>🚀 Cloud-native architecture</p>
</div>
""", unsafe_allow_html=True)