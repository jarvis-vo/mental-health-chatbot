import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from utils.chat_utils import format_chat_history


def init_chat_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "is_responding" not in st.session_state:
        st.session_state.is_responding = False


def display_chat_history():
    for message in st.session_state.chat_history:
        with st.chat_message(
                "user" if isinstance(message, HumanMessage) else "assistant"):
            formatted_content = message.content.replace('\n', '  \n')
            st.markdown(formatted_content, unsafe_allow_html=True)


def handle_user_input(llm_chain):
    user_query = st.chat_input(
        "Share your thoughts...",
        disabled=st.session_state.is_responding,
    )

    if user_query:
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.is_responding = True
        st.rerun()


def handle_ai_response(llm_chain):
    if st.session_state.is_responding and st.session_state.chat_history:
        with st.chat_message("assistant"), st.spinner("Thinking..."):
            ai_response = llm_chain.invoke({
                "chat_history": format_chat_history(st.session_state.chat_history),
                "user_input": st.session_state.chat_history[-1].content
            })

        st.session_state.chat_history.append(AIMessage(content=ai_response))
        st.session_state.is_responding = False
        st.rerun()