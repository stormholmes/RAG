import streamlit as st
from streamlit_option_menu import option_menu

st.markdown("""
    <style>
    .stButton button {
        width: 100%;
        height: 100px;
        font-size: 50px;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("### 🚀 NLP RAG Project - Intelligent Document Chatbot")
st.info("Advanced Retrieval-Augmented Generation System for Document Processing and Conversational AI")

# Main NLP RAG Chatbot button - prominently displayed
st.markdown("### 🎯 Main Application")
if st.button("🚀 NLP RAG Chatbot", key="main_rag_button"):
    st.switch_page("pages/enhanced_rag_chatbot.py")

st.markdown("---")




