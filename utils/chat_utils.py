from typing import List

from langchain_core.messages import HumanMessage, AIMessage


def format_chat_history(chat_history: List) -> str:
    formatted_history = ""
    for message in chat_history[:-1]:  # Exclude the latest message
        if isinstance(message, HumanMessage):
            formatted_history += f"<|user|>\n{message.content}</s>\n"
        elif isinstance(message, AIMessage):
            formatted_history += f"<|assistant|>\n{message.content}</s>\n"
    return formatted_history

def load_css(fpath: str) -> str:
    with open(fpath, "r") as f:
        return f.read()