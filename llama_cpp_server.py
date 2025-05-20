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
    rag_file_path: Optional[str] = None   # <-- RAG 파일 경로 추가!
    rag_query: Optional[str] = None       # <-- RAG용 쿼리 추가!

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
NUM_MODELS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "1"))  # 사용할 모델 인스턴스 수

# --- 간단 RAG 함수 ---
def search_rag_file(rag_file_path, query):
    # 파일 경로에 ~ 있으면 홈 디렉토리로 변환
    rag_file_path = os.path.expanduser(rag_file_path)
    if not rag_file_path or not os.path.isfile(rag_file_path):
        return ""
    with open(rag_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # 유사도 검색 간단 버전: query가 들어간 줄 3개만 추출
    results = [line.strip() for line in lines if query.lower() in line.lower()]
    return "\n".join(results[:3]) if results else ""

# --- 모델 관리 클래스 ---
class ModelManager:
    def __init__(self, num_models):
        self.models = []
        self.model_locks = []  # 각 모델별 락
        self.counter = 0
        self.counter_lock = threading.Lock()  # 카운터용 락
        self.load_models(num_models)
        
    def load_models(self, num_models):
        for i in range(num_models):
            print(f"Loading model instance {i+1}/{num_models}...")
            model = Llama(
                model_path=MODEL_PATH,
                n_threads=THREADS,
                n_ctx=CTX_SIZE,
                n_gpu_layers=GPU_LAYERS,
                low_vram=LOW_VRAM,
                verbose=False,
            )
            self.models.append(model)
            self.model_locks.append(threading.Lock())
            
            # 첫 번째 모델에 대해서만 워밍업 실행
            if i == 0:
                try:
                    print("Warming up model...")
                    _ = model("워밍업", max_tokens=1)
                    print("Model warmed up successfully")
                except Exception as e:
                    print(f"Warning: Warmup failed: {e}")
        
        print(f"All {num_models} model instances loaded successfully")
    
    def get_model(self):
        # 원자적으로 카운터 증가 및 모델 선택
        with self.counter_lock:
            model_idx = self.counter % len(self.models)
            self.counter += 1
            print(f"Selected model instance: {model_idx}")
            return model_idx
    
    def execute(self, model_idx, prompt, max_tokens):
        # 선택된 모델에 대한 락 획득
        with self.model_locks[model_idx]:
            print(f"Executing on model instance {model_idx}")
            return self.models[model_idx](prompt, max_tokens=max_tokens)

# 모델 매니저 인스턴스 생성
model_manager = None

# 각 요청에 대한 세마포어 (동시 요청 제한)
INFERENCE_SEM = threading.Semaphore(MAX_CONCURRENT)

@app.on_event("startup")
def on_startup():
    global model_manager
    print("Initializing model manager...")
    model_manager = ModelManager(NUM_MODELS)
    print("Server startup complete")

@app.get("/healthz")
def healthz():
    return {"status": "ok", "model": MODEL_PATH, "instances": len(model_manager.models) if model_manager else 0}

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

    # RAG 파일에서 내용 검색 (옵션)
    rag_snippet = ""
    if request.rag_file_path and request.rag_query:
        rag_snippet = search_rag_file(request.rag_file_path, request.rag_query)
        if rag_snippet:
            # 프롬프트에 RAG 결과 추가
            full_prompt.append({"role": "system", "content": f"[RAG_RESULT]\n{rag_snippet}"})

    prompt_text = "<bos>"  # 시작 토큰
    system_text = None

    for m in full_prompt:
        role = m["role"] if isinstance(m, dict) else m.role
        content = m["content"] if isinstance(m, dict) else m.content

        if role == "system":
            # Gemma는 system 메시지를 user block 안에 포함시키는 게 일반적
            system_text = content.strip()
        elif role == "user":
            prompt_text += f"<start_of_turn>user\n"
            if system_text:
                prompt_text += system_text + "\n\n"  # system을 user 메시지에 포함
                system_text = None  # system 메시지는 첫 번째 user 메시지에만 포함
            prompt_text += content.strip() + "\n<end_of_turn>\n"
        elif role == "assistant":
            prompt_text += f"<start_of_turn>model\n{content.strip()}\n<end_of_turn>\n"

    # 모델 응답용 자리 남기기
    prompt_text += "<start_of_turn>model\n"

    # 🔒토큰 길이 초과 검사
    prompt_tokens = model_manager.models[0].tokenize(prompt_text.encode("utf-8"))
    total_requested = len(prompt_tokens) + request.max_tokens
    if total_requested > CTX_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"❌ Token limit exceeded: {total_requested} > {CTX_SIZE} (prompt: {len(prompt_tokens)} + max: {request.max_tokens})"
        )

    request_id = uuid.uuid4().hex
    print(f"Request {request_id}: Waiting for semaphore")
    INFERENCE_SEM.acquire()
    try:
        print(f"Request {request_id}: Acquired semaphore")
        # 모델 인스턴스 선택
        model_idx = model_manager.get_model()
        print(f"Request {request_id}: Selected model {model_idx}")
        
        # 선택된 모델로 추론 실행
        result = model_manager.execute(model_idx, prompt_text, request.max_tokens)
        print(f"Request {request_id}: Inference completed with model {model_idx}")

        # 결과 처리
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
            "id": f"chatcmpl-{request_id[:14]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": os.path.basename(MODEL_PATH),
            "model_instance": model_idx,  # 어떤 모델 인스턴스가 처리했는지 추가
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
            f.write(f"\n===== NEW REQUEST {request_id} (MODEL {model_idx}) =====\n")
            f.write("PROMPT:\n" + prompt_text + "\n\n")
            f.write("RAW RESULT:\n" + json.dumps(result, indent=2, default=str) + "\n\n")
            f.write("FINAL RESPONSE:\n" + json.dumps(response, indent=2) + "\n")

        print(f"Request {request_id}: Completed successfully")
        return JSONResponse(content=response)

    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"Request {request_id}: Error during model call: {e}")
        print(traceback_str)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        INFERENCE_SEM.release()
        print(f"Request {request_id}: Released semaphore")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",  # app 모듈의 app 객체 (파일명이 app.py인 경우)
        host="0.0.0.0",
        port=30000,
        log_level="info",
        reload=False
    )
