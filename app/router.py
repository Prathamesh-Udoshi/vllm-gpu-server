import uuid
from fastapi import APIRouter, HTTPException, Depends, Security, Header
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import StreamingResponse

from vllm.sampling_params import SamplingParams

from app.config import settings
from app.schemas import (
    ChatCompletionRequest,
    CompletionRequest,
    ModelList,
    ModelCard
)
from app.engine import llm_engine

router = APIRouter()

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(
    api_key: str = Security(api_key_header),
    authorization: str = Header(None)
):
    """
    Verify API Key if settings.API_KEY is configured.
    Supports either X-API-Key header or Authorization: Bearer <key>.
    """
    if not settings.API_KEY:
        return True  # Authentication disabled

    token = api_key
    if not token and authorization:
        if authorization.startswith("Bearer "):
            token = authorization[7:].strip()
        else:
            token = authorization.strip()

    if token != settings.API_KEY:
        raise HTTPException(
            status_code=401,
            detail={"error": {"message": "Invalid or missing API key.", "type": "authentication_error", "code": 401}}
        )
    return True

@router.get("/v1/models", response_model=ModelList, dependencies=[Depends(verify_api_key)])
async def list_models():
    """List loaded model details in OpenAI compatible format."""
    return ModelList(
        data=[
            ModelCard(id=settings.MODEL_NAME)
        ]
    )

@router.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def create_chat_completion(request: ChatCompletionRequest):
    """
    OpenAI-compatible Chat Completions API.
    Supports both streaming (SSE) and non-streaming responses.
    """
    if not llm_engine.engine:
        raise HTTPException(status_code=503, detail="vLLM engine is initializing or unavailable")

    request_id = f"cmpl-{uuid.uuid4().hex[:12]}"

    messages_dicts = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    prompt = llm_engine.build_prompt_from_messages(messages_dicts)

    stop_sequences = []
    if request.stop:
        if isinstance(request.stop, str):
            stop_sequences = [request.stop]
        elif isinstance(request.stop, list):
            stop_sequences = request.stop

    sampling_params = SamplingParams(
        temperature=request.temperature or 0.7,
        top_p=request.top_p or 1.0,
        max_tokens=request.max_tokens or 512,
        stop=stop_sequences,
        presence_penalty=request.presence_penalty or 0.0,
        frequency_penalty=request.frequency_penalty or 0.0
    )

    if request.stream:
        return StreamingResponse(
            llm_engine.generate_stream(
                request_id=request_id,
                prompt=prompt,
                sampling_params=sampling_params,
                model_name=request.model or settings.MODEL_NAME
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    else:
        return await llm_engine.generate_non_stream(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            model_name=request.model or settings.MODEL_NAME
        )

@router.post("/v1/completions", dependencies=[Depends(verify_api_key)])
async def create_completion(request: CompletionRequest):
    """Standard text completion endpoint."""
    if not llm_engine.engine:
        raise HTTPException(status_code=503, detail="vLLM engine is initializing or unavailable")

    request_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]

    sampling_params = SamplingParams(
        temperature=request.temperature or 0.7,
        top_p=request.top_p or 1.0,
        max_tokens=request.max_tokens or 512
    )

    if request.stream:
        return StreamingResponse(
            llm_engine.generate_stream(
                request_id=request_id,
                prompt=prompt,
                sampling_params=sampling_params,
                model_name=request.model or settings.MODEL_NAME
            ),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"}
        )
    else:
        return await llm_engine.generate_non_stream(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            model_name=request.model or settings.MODEL_NAME
        )
