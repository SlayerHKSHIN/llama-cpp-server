# file: llama_cpp_server.py
import os
import asyncio

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from llama_cpp import Llama

# --- API Key 설정 ---
VALID_API_KEYS = {
    os.getenv("API_KEY_1", ""),
    os.getenv("API_KEY_2", ""),
    os.getenv("API_KEY_3", ""),
}
VALID_API_KEYS.discard("")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key(key: str = Depends(api_key_header)):
    if not key or key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key"
        )
    return key

# --- FastAPI 앱 초기화 ---
app = FastAPI(title="llama-cpp-python Chat API")

# --- 요청/응답 스키마 ---
class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 128

class ChatChoice(BaseModel):
    text: str

class ChatResponse(BaseModel):
    id: str
    choices: list[ChatChoice]

# --- 환경 변수 로드 ---
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.expanduser("~/local_llm_models/gemma-3-27b-it-q4_0.gguf")
)
THREADS    = int(os.getenv("THREADS",  "0"))    # 0 = 자동, CPU Thread 사용 하는 내용
CTX_SIZE   = int(os.getenv("CTX_SIZE", "1024"))
GPU_LAYERS = int(os.getenv("GPU_LAYERS", "40"))
LOW_VRAM   = bool(int(os.getenv("LOW_VRAM", "0"))) # 1=enable, 0=disable, vram이 낮을 경우 CPU를 쓰게 하는 옵션. gpu만 쓰게 하려면 disable 하는 것이 맞음. 

# --- 동시 처리 제한 (세마포어) ---
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "4"))
sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# --- 모델 인스턴스 로드 함수 ---
model: Llama  # 타입 힌트
def reload_model():
    global model
    model = Llama(
        model_path   = MODEL_PATH,
        n_threads    = THREADS,
        n_ctx        = CTX_SIZE,
        n_gpu_layers = GPU_LAYERS,
        low_vram     = LOW_VRAM,
        verbose      = False,
    )
    # 워밍업: GPU 페이지인
    _ = model("워밍업", max_tokens=1)

# --- 서버 시작 시 모델 로드 ---
@app.on_event("startup")
async def on_startup():
    reload_model()

# --- 헬스 체크 엔드포인트 ---
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

# 서버 전체에서 모델 추론 동시 실행 수를 제한할 세마포어
INFERENCE_SEM = asyncio.Semaphore(4)   # 예: 동시에 최대 4개까지 추론

# --- 채팅 엔드포인트: 인증 + 동시성 제어 ---
@app.post(
    "/v1/chat/completions",
    response_model=ChatResponse,
    dependencies=[Depends(get_api_key)]
)
async def chat(request: ChatRequest):
    async with sem:
        loop = asyncio.get_running_loop()
        try:
            # 블로킹 추론은 별도 쓰레드로 분리
            result = await loop.run_in_executor(
                None,
                lambda: model(
                    request.prompt,
                    max_tokens=request.max_tokens
                )
            )
            text = result["choices"][0]["text"]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        return ChatResponse(
            id="llama-" + os.urandom(6).hex(),
            choices=[ChatChoice(text=text)]
        )

# --- 로컬 실행용 진입점 ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "llama_cpp_server:app",
        host="0.0.0.0",
        port=30000,
        log_level="info",
        reload=False
    )
