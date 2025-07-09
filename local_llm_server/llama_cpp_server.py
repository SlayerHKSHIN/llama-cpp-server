import os
import time
import uuid
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from llama_cpp import Llama
import threading
from typing import List, Optional, Union
import traceback
import json
from fastapi import Query

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
    rag_file_path: Optional[str] = None
    rag_query: Optional[str] = None
    rag_candidates: Optional[List[str]] = None

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

# --- RAG 파일에서 후보 리스트 반환 ---
def search_rag_file(rag_file_path, rag_query):
    # 1. 경로 변환 및 로그
    expanded_path = os.path.expanduser(rag_file_path)
    print(f"[RAG] 입력 파일 경로(raw): {rag_file_path}")
    print(f"[RAG] 입력 파일 경로(확장): {expanded_path}")

    # 2. 파일 존재 여부 체크 및 로그
    if not expanded_path or not os.path.isfile(expanded_path):
        print(f"[RAG] 파일이 존재하지 않습니다: {expanded_path}")
        return []

    # 3. JSON 로드 및 에러 처리
    try:
        with open(expanded_path, "r", encoding="utf-8") as f:
            candidates = json.load(f)
        print(f"[RAG] 파일에서 불러온 후보 리스트: {candidates}")
        if not isinstance(candidates, list):
            print(f"[RAG] 경고: 후보 리스트가 list 타입이 아닙니다. 타입: {type(candidates)}")
        return candidates
    except Exception as e:
        print(f"[RAG][ERROR] JSON 파일 로딩 실패: {e}")
        return []

# --- 모델 관리 클래스 ---
class ModelManager:
    def __init__(self, num_models):
        self.models = []
        self.model_locks = []
        self.counter = 0
        self.counter_lock = threading.Lock()
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
                verbose=True,
            )
            self.models.append(model)
            self.model_locks.append(threading.Lock())
            if i == 0:
                try:
                    print("Warming up model...")
                    _ = model("워밍업", max_tokens=1)
                    print("Model warmed up successfully")
                except Exception as e:
                    print(f"Warning: Warmup failed: {e}")
        print(f"All {num_models} model instances loaded successfully")

    def get_model(self):
        with self.counter_lock:
            model_idx = self.counter % len(self.models)
            self.counter += 1
            print(f"Selected model instance: {model_idx}")
            return model_idx

    def execute(self, model_idx, prompt, max_tokens, stream=False):
        with self.model_locks[model_idx]:
            print(f"Executing on model instance {model_idx} (stream={stream})")
            start_time = time.time()
            result = self.models[model_idx](prompt, max_tokens=max_tokens, stream=stream)
            
            # Calculate performance metrics for non-streaming
            if not stream:
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                # Extract token counts
                if isinstance(result, dict):
                    completion_tokens = result.get("usage", {}).get("n_eval", 0)
                    prompt_tokens = result.get("usage", {}).get("n_prompt", 0)
                    
                    # Calculate tokens per second
                    if elapsed_time > 0 and completion_tokens > 0:
                        tokens_per_second = completion_tokens / elapsed_time
                        print(f"Performance Metrics:")
                        print(f"  - Prompt tokens: {prompt_tokens}")
                        print(f"  - Completion tokens: {completion_tokens}")
                        print(f"  - Total time: {elapsed_time:.2f}s")
                        print(f"  - Generation speed: {tokens_per_second:.2f} tokens/s")
                        
                        # Also log to file
                        with open("/tmp/llm_performance.log", "a") as f:
                            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Model {model_idx}: {tokens_per_second:.2f} tok/s, {completion_tokens} tokens in {elapsed_time:.2f}s\n")
            
            return result

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

async def stream_chat_response(model_idx, prompt_text, max_tokens, request_id):
    """Generate streaming response for chat completions"""
    try:
        start_time = time.time()
        token_count = 0
        
        # Get streaming response from model
        stream_generator = model_manager.models[model_idx](
            prompt_text, 
            max_tokens=max_tokens, 
            stream=True
        )
        
        accumulated_text = ""
        for chunk in stream_generator:
            if isinstance(chunk, dict) and "choices" in chunk:
                delta_text = chunk["choices"][0]["text"]
                accumulated_text += delta_text
                token_count += 1  # Approximate token count
                
                # Create SSE formatted chunk
                stream_chunk = {
                    "id": f"chatcmpl-{request_id[:14]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": os.path.basename(MODEL_PATH),
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "content": delta_text
                        },
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(stream_chunk)}\n\n"
        
        # Calculate streaming performance
        elapsed_time = time.time() - start_time
        if elapsed_time > 0 and token_count > 0:
            tokens_per_second = token_count / elapsed_time
            print(f"Streaming Performance Metrics:")
            print(f"  - Approximate tokens generated: {token_count}")
            print(f"  - Total time: {elapsed_time:.2f}s")
            print(f"  - Generation speed: {tokens_per_second:.2f} tokens/s")
            
            with open("/tmp/llm_performance.log", "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Model {model_idx} (streaming): {tokens_per_second:.2f} tok/s, {token_count} tokens in {elapsed_time:.2f}s\n")
        
        # Send final chunk with finish_reason
        final_chunk = {
            "id": f"chatcmpl-{request_id[:14]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": os.path.basename(MODEL_PATH),
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "server_error",
                "code": 500
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"

@app.post(
    "/v1/chat/completions",
    dependencies=[Depends(get_api_key)]
)
async def chat(request: ChatRequest):
    # -------- 프롬프트 조립 -----------
    rag_candidates = []
    if request.rag_candidates is not None:
        rag_candidates = request.rag_candidates
        print("RAG 후보 리스트 (rag_candidates로부터):", rag_candidates)
    elif request.rag_file_path and request.rag_query:
        rag_candidates = search_rag_file(request.rag_file_path, request.rag_query)
        print("RAG 후보 리스트 (파일로부터):", rag_candidates)
        print("입력 이름:", request.rag_query)

    # 시스템 프롬프트 조립
    base_system_prompt = request.system_prompt or ""
    rag_prompt = ""
    if rag_candidates:
        rag_prompt = (
            f"\n\n[NAME RAG]\n"
            f"User input: {request.rag_query}\n"
            f"Candidates: {json.dumps(rag_candidates, ensure_ascii=False)}\n"
            f"INSTRUCTION: "
            f"You must search the candidate list above for a name that is at least 80% similar to the user input. "
            f"If you find such a name, output the exact name from the candidate list. "
            f"If there is no suitable name, output fail. "
            f"Never guess, invent, or modify a name. "
            f"Only return the candidate name or fail. No explanation, no formatting."
        )
    if rag_prompt:
        system_text = base_system_prompt + rag_prompt
    else:
        system_text = base_system_prompt

    full_prompt = []
    if system_text:
        full_prompt.append({"role": "system", "content": system_text})
    full_prompt += request.messages
    if request.user_prompt_input:
        full_prompt.append({"role": "user", "content": request.user_prompt_input})

    # --------- 모델 프롬프트 생성 -----------
    prompt_text = "<bos>"  # 시작 토큰
    sys_msg = None

    for m in full_prompt:
        role = m["role"] if isinstance(m, dict) else m.role
        content = m["content"] if isinstance(m, dict) else m.content

        if role == "system":
            sys_msg = content.strip()
        elif role == "user":
            prompt_text += f"<start_of_turn>user\n"
            if sys_msg:
                prompt_text += sys_msg + "\n\n"
                sys_msg = None
            prompt_text += content.strip() + "\n<end_of_turn>\n"
        elif role == "assistant":
            prompt_text += f"<start_of_turn>model\n{content.strip()}\n<end_of_turn>\n"

    prompt_text += "<start_of_turn>model\n"

    # 토큰 길이 초과 검사
    prompt_tokens = model_manager.models[0].tokenize(prompt_text.encode("utf-8"))
    total_requested = len(prompt_tokens) + request.max_tokens
    if total_requested > CTX_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"❌ Token limit exceeded: {total_requested} > {CTX_SIZE} (prompt: {len(prompt_tokens)} + max: {request.max_tokens})"
        )

    request_id = uuid.uuid4().hex
    print(f"Request {request_id}: Waiting for semaphore (stream={request.stream})")
    
    # Handle streaming response
    if request.stream:
        INFERENCE_SEM.acquire()
        try:
            print(f"Request {request_id}: Acquired semaphore for streaming")
            model_idx = model_manager.get_model()
            print(f"Request {request_id}: Selected model {model_idx} for streaming")
            
            # Create generator wrapper that releases semaphore when done
            async def stream_with_cleanup():
                try:
                    async for chunk in stream_chat_response(model_idx, prompt_text, request.max_tokens, request_id):
                        yield chunk
                finally:
                    INFERENCE_SEM.release()
                    print(f"Request {request_id}: Released semaphore after streaming")
            
            # Return streaming response
            return StreamingResponse(
                stream_with_cleanup(),
                media_type="text/event-stream"
            )
        except Exception as e:
            INFERENCE_SEM.release()
            print(f"Request {request_id}: Error setting up stream: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Handle non-streaming response
    INFERENCE_SEM.acquire()
    try:
        print(f"Request {request_id}: Acquired semaphore")
        model_idx = model_manager.get_model()
        print(f"Request {request_id}: Selected model {model_idx}")

        result = model_manager.execute(model_idx, prompt_text, request.max_tokens, stream=False)
        print(f"Request {request_id}: Inference completed with model {model_idx}")

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
            "model_instance": model_idx,
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

@app.get(
    "/v1/honeybadger",
    response_class=JSONResponse,
    dependencies=[Depends(get_api_key)]
)
async def honeybadger_api(
    u: str = Query(..., description="Requested URL identifier, e.g. page or screen name"),
    c: str = Query(None, description="Supplemental context as JSON string, nullable"),
    q: str = Query(..., description="Question prompt")
):
    request_id = uuid.uuid4().hex
    print(f"Honeybadger request {request_id}: url={u}, context={c}, question={q}")

    # Parse context JSON if provided
    context_data = {}
    if c:
        try:
            context_data = json.loads(c)
            print(f"Parsed context JSON: {context_data}")
        except json.JSONDecodeError:
            print(f"Warning: Failed to parse context JSON: {c}")
            # Optionally, you could raise HTTPException here to reject bad JSON
            # raise HTTPException(status_code=400, detail="Invalid JSON in context parameter")

    # Compose system prompt including URL identifier and parsed context
    system_prompt = (
        f"I am Honeybadger, your confident and slightly cheeky assistant. "
        f"You asked a question related to the page '{u}'. "
        f"The supplemental context I have is: {json.dumps(context_data, ensure_ascii=False)}. "
        f"Answer clearly and with charm."
    )

    prompt_text = "<bos>"
    prompt_text += f"<start_of_turn>system\n{system_prompt}\n<end_of_turn>\n"
    prompt_text += f"<start_of_turn>user\n{q}\n<end_of_turn>\n"
    prompt_text += "<start_of_turn>model\n"

    # Token length check
    prompt_tokens = model_manager.models[0].tokenize(prompt_text.encode("utf-8"))
    total_requested = len(prompt_tokens) + 128  # max_tokens
    if total_requested > CTX_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Token limit exceeded: {total_requested} > {CTX_SIZE}"
        )

    INFERENCE_SEM.acquire()
    try:
        model_idx = model_manager.get_model()
        result = model_manager.execute(model_idx, prompt_text, max_tokens=128)

        if isinstance(result, dict):
            if "choices" in result and isinstance(result["choices"], list):
                text = result["choices"][0]["text"]
            elif "text" in result:
                text = result["text"]
            else:
                raise ValueError("Unexpected response format")
        else:
            raise ValueError("Model returned non-dict result")

        response = {
            "id": f"honeybadger-{request_id[:14]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": os.path.basename(MODEL_PATH),
            "model_instance": model_idx,
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
        }
        print(f"Honeybadger request {request_id} completed")
        return JSONResponse(content=response)

    except Exception as e:
        print(f"Honeybadger request {request_id} error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        INFERENCE_SEM.release()
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=30000,
        log_level="info",
        reload=False
    )
