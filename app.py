import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from components.header import render_header
from components.sidebar import render_sidebar
from configs.prompts import get_chat_prompt
from models.mental_chatbot import MentalChatbot
from utils.chat_utils import format_chat_history, load_css

# Set page config
st.set_page_config(
    page_title="MindEase",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load and apply CSS
css = load_css('assets/css/styles.css')
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Render components
render_header()
render_sidebar()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "is_responding" not in st.session_state:
    st.session_state.is_responding = False

if "llm_chain" not in st.session_state:
    chat_prompt = get_chat_prompt()
    llm = MentalChatbot(max_new_tokens=24)
    st.session_state.llm_chain = chat_prompt | llm | StrOutputParser()

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
        formatted_content = message.content.replace('\n', '  \n')
        st.markdown(formatted_content, unsafe_allow_html=True)

# Handle user input, disable chat input if model is responding
user_query = st.chat_input(
    "Share your thoughts...",
    disabled=st.session_state.is_responding,
    key="chat_input"
)

if user_query:
    with st.chat_message("user"):
        pass

    st.session_state.chat_history.append(HumanMessage(content=user_query))
    formatted_history = format_chat_history(st.session_state.chat_history)

    # Set responding state to True
    st.session_state.is_responding = True
    st.rerun()

# Handle AI response in a separate block
if st.session_state.is_responding and st.session_state.chat_history:
    with st.chat_message("assistant"), st.spinner("Thinking..."):
        ai_response = st.session_state.llm_chain.invoke({
            "chat_history": format_chat_history(st.session_state.chat_history),
            "user_input": st.session_state.chat_history[-1].content
        })
        st.session_state.chat_history.append(AIMessage(content=ai_response))

    # Reset responding state
    st.session_state.is_responding = False
    st.rerun()