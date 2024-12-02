from langchain.prompts import PromptTemplate

MENTAL_HEALTH_TEMPLATE = """
<|system|>
You are a helpful mental health chatbot. Provide supportive and empathetic responses.</s>
{chat_history}
<|user|>
{user_input}</s>
<|assistant|>
"""

def get_chat_prompt() -> PromptTemplate:
    return PromptTemplate(
        template=MENTAL_HEALTH_TEMPLATE,
        input_variables=["chat_history", "user_input"]
    ) 