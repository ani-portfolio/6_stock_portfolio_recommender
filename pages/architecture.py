import streamlit as st
from PIL import Image
import os
import sys
import base64
from io import BytesIO

sys.path.append('..')

st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center; color: black;'>SYSTEM ARCHITECTURE</h1>", unsafe_allow_html=True)

try:
    image_path = "media/architecture.png"
    if os.path.exists(image_path):
        architecture_image = Image.open(image_path)
        
        # Convert image to base64 for HTML embedding with maximum quality
        buffer = BytesIO()
        architecture_image.save(buffer, format='PNG', optimize=False, quality=100)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        # Display with high-quality HTML rendering
        st.markdown(
            f'''
            <style>
            .high-quality-image-container {{
                display: flex;
                justify-content: center;
                margin: 20px 0;
                border: 2px solid #ddd;
                border-radius: 10px;
                padding: 10px;
                background: #f9f9f9;
            }}
            
            .high-quality-image {{
                max-width: 100%;
                height: auto;
                image-rendering: -webkit-optimize-contrast;
                image-rendering: -moz-crisp-edges;
                image-rendering: crisp-edges;
                image-rendering: pixelated;
                -ms-interpolation-mode: nearest-neighbor;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                border-radius: 5px;
            }}
            </style>
            
            <div class="high-quality-image-container">
                <img src="data:image/png;base64,{img_str}" 
                     class="high-quality-image"
                     alt="System Architecture Diagram">
            </div>
            ''',
            unsafe_allow_html=True
        )
        
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