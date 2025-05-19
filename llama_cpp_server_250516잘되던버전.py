import os
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from llama_cpp import Llama
import asyncio
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
app = FastAPI(title="llama-cpp Chat API")

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 128

class ChatChoice(BaseModel):
    text: str

class ChatResponse(BaseModel):
    id: str
    choices: list[ChatChoice]

# 동시 inference 제한: **1**
MAX_CONCURRENT = 1
INFERENCE_SEM = asyncio.Semaphore(MAX_CONCURRENT)

# 모델 인스턴스 + 로드 락
model: Llama | None = None
model_lock = threading.Lock()

def get_model() -> Llama:
    global model
    if model is None:
        with model_lock:
            if model is None:
                model = Llama(
                    model_path   = os.environ["MODEL_PATH"],
                    n_threads    = int(os.getenv("THREADS","0")),
                    n_ctx        = int(os.getenv("CTX_SIZE","1024")),
                    n_gpu_layers = int(os.getenv("GPU_LAYERS","40")),
                    low_vram     = bool(int(os.getenv("LOW_VRAM","0"))),
                    verbose      = False,
                )
                # 워밍업
                try:
                    _ = model("워밍업", max_tokens=1)
                except:
                    pass
    return model  # type: ignore

@app.get("/healthz")
def healthz():
    return {"status":"ok"}

@app.post(
    "/v1/chat/completions",
    response_model=ChatResponse,
    dependencies=[Depends(get_api_key)]
)
async def chat(request: ChatRequest):
    # “한 번에 하나” 만 진입
    async with INFERENCE_SEM:
        try:
            loop = asyncio.get_running_loop()
            # 실제 inference 는 별 쓰레드에서 실행
            result = await loop.run_in_executor(
                None,
                lambda: get_model()(request.prompt, max_tokens=request.max_tokens)
            )
            text = result["choices"][0]["text"]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(
        id="llama-" + os.urandom(6).hex(),
        choices=[ChatChoice(text=text)]
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
