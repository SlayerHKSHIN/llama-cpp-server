# file: llama_cpp_server.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama

app = FastAPI(title="llama-cpp-python Chat API")

# --- Request & Response schemas ---
class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int = 128

class ChatChoice(BaseModel):
    text: str

class ChatResponse(BaseModel):
    id: str
    choices: list[ChatChoice]

# --- Load config from env ---
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.expanduser("~/local_llm_models/gemma-3-27b-it-q4_0.gguf")
)
THREADS    = int(os.getenv("THREADS",  "0"))    # 0 = 자동 스레드 수
CTX_SIZE   = int(os.getenv("CTX_SIZE", "1024"))
GPU_LAYERS = int(os.getenv("GPU_LAYERS", "40"))
LOW_VRAM   = bool(int(os.getenv("LOW_VRAM", "0")))  # 1 = enable, 0 = disable

# --- Initialize model once at startup ---
model = Llama(
    model_path   = MODEL_PATH,
    n_threads    = THREADS,
    n_ctx        = CTX_SIZE,
    n_gpu_layers = GPU_LAYERS,
    low_vram     = LOW_VRAM,
    verbose      = False,
)

# --- Health check ---
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

# --- Chat endpoint ---
@app.post("/v1/chat/completions", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        result = model(
            request.prompt,
            max_tokens=request.max_tokens,
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
        port=8000,
        log_level="info",
        reload=True
    )
