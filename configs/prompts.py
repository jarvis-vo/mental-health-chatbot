from langchain.prompts import PromptTemplate

CHAT_PROMPT_TEMPLATE = """
<|system|>
You are a helpful mental health chatbot. Provide supportive and empathetic responses.</s>
{chat_history}
<|user|>
{user_input}</s>
<|assistant|>
"""

def get_chat_prompt() -> PromptTemplate:
    return PromptTemplate(
        template=CHAT_PROMPT_TEMPLATE,
        input_variables=["chat_history", "user_input"]
    ) 