from typing import Any, List, Optional

import torch
from langchain_core.language_models.llms import LLM
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


class MentalChatbot(LLM, BaseModel):
    model_name: str = Field(default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    max_new_tokens: int = Field(default=256)
    tokenizer: Any = Field(default=None, exclude=True)
    model: Any = Field(default=None, exclude=True)

    def __init__(self,
                 model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                 max_new_tokens: int = 256,
                 **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16, device_map="auto")

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = self.tokenizer(prompt,
                                return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=20,
            top_p=0.95,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()

    @property
    def _llm_type(self) -> str:
        return "mental_chatbot"
