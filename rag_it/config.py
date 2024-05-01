import os

MODELS_PATH = os.getenv("MODELS_PATH", "models")
LLAMA2_7B_PATH = MODELS_PATH + "/7b-ggml-model-f32_q4_0.bin"
MISTRAL_7B_PATH = MODELS_PATH + "/ggml-model-q4_0.gguf"

DATA_PATH = os.getenv("DATA_PATH", "data")

INDICES_PATH = os.getenv("INDICES_PATH", "indices")
CHROMA_INDEX_PATH = INDICES_PATH + "/chroma_db"
