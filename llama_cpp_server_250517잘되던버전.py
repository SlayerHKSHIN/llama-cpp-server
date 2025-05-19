import os
import asyncio
import threading
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from typing import Literal, Union

from llama_cpp import Llama

# --- API Key 설정 ---
VALID_API_KEYS = {
    os.getenv("API_KEY_1", ""),
    os.getenv("API_KEY_2", ""),
    os.getenv("API_KEY_3", ""),
} - {""}
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(key: str = Depends(api_key_header)):
    if key not in VALID_API_KEYS:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
    return key

# --- FastAPI 앱 ---
app = FastAPI(title="llama-cpp Chat API")

# --- 입력 데이터 구조 ---
class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    prompt: Union[str, None] = None
    messages: Union[list[Message], None] = None
    max_tokens: int = 128

# --- 출력 구조 ---
class ChatChoice(BaseModel):
    text: str

class ChatResponse(BaseModel):
    id: str
    choices: list[ChatChoice]

# --- 동시 inference 제한 ---
MAX_CONCURRENT = 1
INFERENCE_SEM = asyncio.Semaphore(MAX_CONCURRENT)

# --- 모델 로딩 ---
model: Llama | None = None
model_lock = threading.Lock()

def get_model() -> Llama:
    global model
    if model is None:
        with model_lock:
            if model is None:
                model = Llama(
                    model_path   = os.environ["MODEL_PATH"],
                    n_threads    = int(os.getenv("THREADS", "0")),
                    n_ctx        = int(os.getenv("CTX_SIZE", "1024")),
                    n_gpu_layers = int(os.getenv("GPU_LAYERS", "40")),
                    low_vram     = bool(int(os.getenv("LOW_VRAM", "0"))),
                    verbose      = False,
                )
                try:
                    _ = model("워밍업", max_tokens=1)
                except:
                    pass
    return model  # type: ignore

# --- messages → prompt 변환 ---
def convert_messages_to_prompt(messages: list[Message]) -> str:
    prompt = ""
    for msg in messages:
        role = msg.role
        content = msg.content.strip()
        if role == "system":
            prompt += f"[System]\n{content}\n\n"
        elif role == "user":
            prompt += f"[User]\n{content}\n\n"
        elif role == "assistant":
            prompt += f"[Assistant]\n{content}\n\n"
    return prompt.strip()

# --- 상태 확인 ---
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

# --- 챗 요청 처리 ---
@app.post(
    "/v1/chat/completions",
    response_model=ChatResponse,
    dependencies=[Depends(get_api_key)]
)
async def chat(request: ChatRequest):
    async with INFERENCE_SEM:
        try:
            if request.messages:
                final_prompt = convert_messages_to_prompt(request.messages)
            elif request.prompt:
                final_prompt = request.prompt
            else:
                raise HTTPException(status_code=400, detail="Either 'prompt' or 'messages' is required.")

            loop = asyncio.get_running_loop()
            raw = await loop.run_in_executor(
                None,
                lambda: get_model()(final_prompt, max_tokens=request.max_tokens)
            )

            # llama.cpp는 문자열만 반환할 수 있음 → 구조화된 JSON 생성
            if isinstance(raw, str):
                text = raw
            elif isinstance(raw, dict) and "choices" in raw:
                text = raw["choices"][0]["text"]
            else:
                raise HTTPException(status_code=500, detail=f"Unexpected model output: {raw}")

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(
        id="llama-" + os.urandom(6).hex(),
        choices=[ChatChoice(text=text)]
    )

# --- 직접 실행용 ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "llama_cpp_server:app",
        host="0.0.0.0",
        port=30000,
        log_level="info",
        reload=False
    )
