from typing import Any, Optional

import streamlit as st
import torch
from langchain.prompts import PromptTemplate
from langchain_core.language_models.llms import LLM
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoModelForCausalLM, AutoTokenizer


# Set page config
st.set_page_config(
    page_title="Mental Health Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Adjust default Streamlit header */
    header[data-testid="stHeader"] {
        background-color: transparent;
    }

    /* Adjust header */
    .header {
        position: fixed;
        top: -8px;
        background-color: #0E1117;
        width: 100%;
        height: 80px;
        z-index: 999;
        display: flex;
        align-items: flex-end;
    }

    /* Center and limit width of chat messages */
    .stChatMessage {
        width: 800px;
        margin: 0 auto;
    }

    /* Style chat input */
    .stChatInput {
        width: 850px !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }

    /* Target the parent container of the chat input */
    .stChatInput > div {
        width: 100% !important;
    }
</style>
<div class="header">
    <h3>Rem-sama ❤️‍🔥</h3>
</div>
""", unsafe_allow_html=True)


# Update the sidebar styling
with st.sidebar:
    st.header("💭 **New Chat**")

    if st.button("Start New Chat", type="primary", use_container_width=True, icon="💭"):
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("---")
    
    
    st.header("ℹ️ **About**")
    st.markdown("""
    This AI assistant is designed to:
    - Provide a supportive conversation, empathetic and non-judgmental.
    - Offer a safe space to share thoughts, feelings, and any concerns.
    - Help you getting through difficult times.
    
    **Note:** This is not a replacement for professional mental health care.
    """)


# Custom LLM class for local TinyLlama
class MentalChatbot(LLM):
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer: Any = None
    model: Any = None

    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
    def _call(self, prompt: str, stop: Optional[list[str]] = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
        )
        
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()
    
    @property
    def _llm_type(self) -> str:
        return "mental_chatbot"

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "llm_chain" not in st.session_state:
    # Create a prompt template that includes conversation history
    prompt_template = """
    <|system|>
    You are a helpful mental health chatbot. Provide supportive and empathetic responses.</s>
    {chat_history}
    <|user|>
    {user_input}</s>
    <|assistant|>
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["chat_history", "user_input"]
    )
    llm = MentalChatbot()
    st.session_state.llm_chain = prompt | llm | StrOutputParser()

# Function to format chat history
def format_chat_history(chat_history):
    formatted_history = ""
    for message in chat_history[:-1]:  # Exclude the latest message
        if isinstance(message, HumanMessage):
            formatted_history += f"<|user|>\n{message.content}</s>\n"
        elif isinstance(message, AIMessage):
            formatted_history += f"<|assistant|>\n{message.content}</s>\n"
    return formatted_history

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Update the user input handling section
if user_query := st.chat_input("Share your thoughts..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Add user message to chat history
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    # Format the chat history
    formatted_history = format_chat_history(st.session_state.chat_history)

    with st.chat_message("assistant"), st.spinner("Thinking..."):
        ai_response = st.session_state.llm_chain.invoke(
            {"chat_history": formatted_history, "user_input": user_query}
        )

    # Add AI response to chat history
    st.session_state.chat_history.append(AIMessage(content=ai_response))

    # Force streamlit to rerun and display the updated chat history
    st.rerun()