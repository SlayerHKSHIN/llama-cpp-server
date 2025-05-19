import os
import time
import uuid
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from llama_cpp import Llama
import threading

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
app = FastAPI(title="llama-cpp-python Chat API")

# --- 메시지 구조 ---
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]
    max_tokens: int = 128

class ChoiceMessage(BaseModel):
    role: str
    content: str

class ChatChoice(BaseModel):
    index: int
    message: ChoiceMessage
    finish_reason: str = "stop"

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: Usage

# --- 환경 변수 로드 ---
MODEL_PATH = os.environ["MODEL_PATH"]
THREADS    = int(os.getenv("THREADS", "0"))
CTX_SIZE   = int(os.getenv("CTX_SIZE", "1024"))
GPU_LAYERS = int(os.getenv("GPU_LAYERS", "40"))
LOW_VRAM   = bool(int(os.getenv("LOW_VRAM", "0")))
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_REQUESTS", "1"))
INFERENCE_SEM = threading.Semaphore(MAX_CONCURRENT)

# --- 전역 모델 인스턴스 로드 ---
model: Llama
def load_model():
    global model
    model = Llama(
        model_path   = MODEL_PATH,
        n_threads    = THREADS,
        n_ctx        = CTX_SIZE,
        n_gpu_layers = GPU_LAYERS,
        low_vram     = LOW_VRAM,
        verbose      = False,
    )
    # 워밍업
    try:
        _ = model("워밍업", max_tokens=1)
    except:
        pass

@app.on_event("startup")
def on_startup():
    load_model()

@app.get("/healthz")
def healthz():
    return {"status": "ok", "model": MODEL_PATH}

@app.post(
    "/v1/chat/completions",
    response_model=ChatResponse,
    dependencies=[Depends(get_api_key)]
)
def chat(request: ChatRequest):
    prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
    INFERENCE_SEM.acquire()
    try:
        result = model(prompt, max_tokens=request.max_tokens)
        text = result["choices"][0]["text"]
        usage = Usage(
            prompt_tokens=result.get("prompt_tokens", 0),
            completion_tokens=result.get("completion_tokens", 0),
            total_tokens=result.get("total_tokens", 0),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        INFERENCE_SEM.release()

    return ChatResponse(
        id="chatcmpl-" + uuid.uuid4().hex[:14],
        created=int(time.time()),
        model=os.path.basename(MODEL_PATH),
        choices=[ChatChoice(
            index=0,
            message=ChoiceMessage(role="assistant", content=text.strip())
        )],
        usage=usage
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "llama_cpp_server:app",
        host="0.0.0.0",
        port=30000,
        log_level="info",
        reload=False
    )
