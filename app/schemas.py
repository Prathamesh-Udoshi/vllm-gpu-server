import time
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, validator

# --- Request Schemas ---

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the author (system, user, assistant)")
    content: str = Field(..., description="Contents of the message")

    @validator('role')
    def validate_role(cls, v):
        if v not in ('system', 'user', 'assistant', 'function', 'tool'):
            raise ValueError(f"Invalid message role: {v}")
        return v

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use")
    messages: List[ChatMessage] = Field(..., description="A list of messages comprising the conversation so far")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(512, ge=1)
    stream: Optional[bool] = Field(False, description="If set, partial message deltas will be sent as SSE")
    stop: Optional[Union[str, List[str]]] = Field(None, description="Up to 4 sequences where the API will stop generating tokens")
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)

class CompletionRequest(BaseModel):
    model: str = Field(...)
    prompt: Union[str, List[str]] = Field(...)
    max_tokens: Optional[int] = Field(512, ge=1)
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    stream: Optional[bool] = Field(False)
    stop: Optional[Union[str, List[str]]] = Field(None)

# --- Response Schemas ---

class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop"

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time()*1000)}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo

class ChoiceDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None

class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChoiceDelta
    finish_reason: Optional[str] = None

class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "vllm-platform"

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]

# --- Error Schemas ---

class ErrorDetail(BaseModel):
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[Union[int, str]] = None

class ErrorResponse(BaseModel):
    error: ErrorDetail
