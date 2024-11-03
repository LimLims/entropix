# File: entropix/prompts.py

from typing import Any, Dict, Literal, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

role_to_header = {
    "system": "user",
    "user": "assistant", 
    "assistant": "user",
}

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1)

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., min_length=1)
    messages: List[Message] = Field(..., min_items=1)
    temperature: Optional[float] = Field(default=1.0, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=2048, ge=1, le=2048)  # Updated for SmolLM context length
    stream: Optional[bool] = Field(default=True)

def generate_chat_prompt(request: ChatCompletionRequest) -> str:
    """Generate SmolLM chat format prompt."""
    prompt = "<|begin_of_text|><|start_header_id|>user\n"
    
    for message in request.messages:
        prompt += f"{message.content}<|im_end|>\n"
        if message != request.messages[-1]:
            prompt += f"<|im_start|>{message.role}\n"
    
    prompt += "<|im_end|>\n<|im_start|>assistant\n"
    return prompt