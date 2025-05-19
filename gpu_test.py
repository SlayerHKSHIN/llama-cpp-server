# gpu_test.py
import os
from llama_cpp import Llama

path = os.path.expanduser("~/local_llm_models/gemma-3-27b-it-q4_0.gguf")
m = Llama(
    model_path=path,
    n_threads=4,
    n_gpu_layers=5,
    low_vram=False,
    verbose=True
)

print(m("GPU 테스트", max_tokens=1))
