import streamlit as st

st.markdown("""<h1 style='text-align: center; color: black;'>LINKS</h1>""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📂 GitHub
    [**Stock Recommender System**](https://github.com/ani-portfolio/6_stock_portfolio_recommender)
    
    Access the complete source code, documentation, and project files for this RAG-LLM powered stock recommendation system.
    """)

with col2:
    st.markdown("""
    ### 🌟 Portfolio
    [**View My Portfolio**](https://www.datascienceportfol.io/ani_dharmarajan)
    
    Explore other data science and ML projects showcasing various technologies and techniques.
    """)

with col3:
    st.markdown("""
    ### 🤝 Connect
    [**LinkedIn**](https://www.linkedin.com/in/ani-dharmarajan/?originalSubdomain=ca)
    
    Connect with me on LinkedIn to discuss data science, machine learning, and career opportunities.
    """)

st.divider()

st.markdown("""
<div style='text-align: center; color: #666;'>
<p> Created by Ani Dharmarajan </p>
</div>
""", unsafe_allow_html=True)