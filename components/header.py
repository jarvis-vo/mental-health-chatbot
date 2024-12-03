import streamlit as st


def render_header():
    st.markdown("""
    <div class="header">
        <h3>MindEase</h3>
    </div>
    """, unsafe_allow_html=True)
