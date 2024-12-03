import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from components.header import render_header
from components.sidebar import render_sidebar
from components.chat import (
    init_chat_state,
    display_chat_history,
    handle_user_input,
    handle_ai_response,
)
from configs.prompts import get_chat_prompt
from models.mental_chatbot import MentalChatbot
from utils.chat_utils import format_chat_history, load_css

# Set page config
st.set_page_config(
    page_title="MindEase",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load and apply CSS
css = load_css('assets/css/styles.css')
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Render components
render_header()
render_sidebar()

# Initialize chat state
init_chat_state()

# Initialize LLM chain if not exists
if "llm_chain" not in st.session_state:
    chat_prompt = get_chat_prompt()
    llm = MentalChatbot(
        model_name="unsloth/Llama-3.2-3B-Instruct",
        max_new_tokens=256,
    )
    st.session_state.llm_chain = chat_prompt | llm | StrOutputParser()

# Display chat interface
display_chat_history()
handle_user_input(st.session_state.llm_chain)
handle_ai_response(st.session_state.llm_chain)
