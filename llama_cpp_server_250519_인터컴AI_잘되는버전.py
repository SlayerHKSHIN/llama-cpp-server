import os
import time
import uuid
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from llama_cpp import Llama
import threading
from typing import List, Optional, Union
import traceback
import json


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

class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: dict

class Tool(BaseModel):
    type: str
    function: ToolFunction

class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: int = 128
    temperature: Optional[float] = 0.7
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, dict]] = None
    stream: Optional[bool] = False
    logprobs: Optional[bool] = False
    conversation_rounds: int = Field(default=100)
    system_prompt: Optional[str] = None
    user_prompt_input: Optional[str] = None

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
    choices: List[ChatChoice]
    usage: Usage

# --- 환경 변수 로드 ---
MODEL_PATH = os.environ["MODEL_PATH"]
THREADS    = int(os.getenv("THREADS", "0"))
CTX_SIZE   = int(os.getenv("CTX_SIZE", "4096"))
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
    response_class=JSONResponse,
    dependencies=[Depends(get_api_key)]
)
async def chat(request: ChatRequest):
    full_prompt = []
    if request.system_prompt:
        full_prompt.append({"role": "system", "content": request.system_prompt})
    full_prompt += request.messages
    if request.user_prompt_input:
        full_prompt.append({"role": "user", "content": request.user_prompt_input})

    prompt_text = "<bos>"  # 시작 토큰

    for m in full_prompt:
        role = m["role"] if isinstance(m, dict) else m.role
        content = m["content"] if isinstance(m, dict) else m.content

        if role == "system":
            # Gemma는 system 메시지를 user block 안에 포함시키는 게 일반적
            system_text = content.strip()
        elif role == "user":
            prompt_text += f"<start_of_turn>user\n"
            if 'system_text' in locals():
                prompt_text += system_text + "\n\n"  # system을 user 메시지에 포함
            prompt_text += content.strip() + "\n<end_of_turn>\n"
        elif role == "assistant":
            prompt_text += f"<start_of_turn>model\n{content.strip()}\n<end_of_turn>\n"

    # 모델 응답용 자리 남기기
    prompt_text += "<start_of_turn>model\n"

    INFERENCE_SEM.acquire()
    try:
        result = model(prompt_text, max_tokens=request.max_tokens)

        # result is expected to be a dict with 'choices' or a dict with 'text'
        if isinstance(result, dict):
            if "choices" in result and isinstance(result["choices"], list):
                text = result["choices"][0]["text"]
            elif "text" in result:
                text = result["text"]
            else:
                raise ValueError("Unexpected response format: missing 'choices' or 'text'")
        else:
            raise ValueError("Model returned non-dict result")

        response = {
            "id": "chatcmpl-" + uuid.uuid4().hex[:14],
            "object": "chat.completion",
            "created": int(time.time()),
            "model": os.path.basename(MODEL_PATH),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text.strip()
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": result.get("prompt_tokens", 0),
                "completion_tokens": result.get("completion_tokens", 0),
                "total_tokens": result.get("total_tokens", 0)
            }
        }

        # 디버깅 로그 저장
        with open("/tmp/llm_log.txt", "a") as f:
            f.write("\n===== NEW REQUEST =====\n")
            f.write("PROMPT:\n" + prompt_text + "\n\n")
            f.write("RAW RESULT:\n" + json.dumps(result, indent=2, default=str) + "\n\n")
            f.write("FINAL RESPONSE:\n" + json.dumps(response, indent=2) + "\n")

        return JSONResponse(content=response)

    except Exception as e:
        traceback_str = traceback.format_exc()
        print("Error during model call:", traceback_str)
        raise HTTPException(status_code=500, detail=traceback_str)
    finally:
        INFERENCE_SEM.release()



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "llama_cpp_server:app",
        host="0.0.0.0",
        port=30000,
        log_level="info",
        reload=False
    )
