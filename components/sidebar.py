import streamlit as st


def render_sidebar():
    with st.sidebar:
        st.header("💭 **New Chat**")
        if st.button("Start New Chat",
                    type="primary",
                    use_container_width=True,
                    icon="💭"):
            st.session_state.chat_history = []
            st.session_state.is_responding = False
            st.rerun()

        st.header("ℹ️ **About**")
        st.markdown("""
        This AI assistant is designed to:
        - Provide a supportive conversation, empathetic and non-judgmental.
        - Offer a safe space to share thoughts, feelings, and any concerns.
        - Help you getting through difficult times.
        
        **Note:** This is not a replacement for professional mental health care.
        """)
