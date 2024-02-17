from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from rag_it.model import Llm
from rag_it.config import LLAMA2_7B



llm = Llm(model_path=LLAMA2_7B, temperature=0.0)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = llm.model
Settings.embed_model = embed_model

documents = SimpleDirectoryReader("data").load_data()

index = VectorStoreIndex(documents, show_progress=True)

query_engine = index.as_query_engine()

response = query_engine.query("Who wrote the Lord of The Rings?")
print(response)