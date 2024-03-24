from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from rag_it.config import DATA_PATH
import os


class Engine:
    def __init__(self, llm, embed_model, data_path, collection_name) -> None:
        self.llm = llm
        self.embed_model = embed_model
        self._settings = Settings
        self._init_settings()
        self.index = self._load_index(data_directory=data_path, collection_name=collection_name)

    def _init_settings(self):
        self._settings.llm = self.llm.model
        self._settings.embed_model = self.embed_model.model

    def _init_storage_context(self, collection_name):
        chroma_client = chromadb.EphemeralClient()
        chroma_collection = chroma_client.create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return storage_context

    def _load_data(self, data_directory):
        documents = SimpleDirectoryReader(os.path.join(DATA_PATH, data_directory)).load_data()
        return documents

    def _load_index(self, data_directory, collection_name):
        documents = self._load_data(data_directory=data_directory)
        storage_context = self._init_storage_context(collection_name=collection_name)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)
        return index

    def query(self, input):
        query_engine = self.index.as_query_engine()
        response = query_engine.query(input)
        return response
