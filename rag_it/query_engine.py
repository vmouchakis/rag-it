from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from rag_it.config import DATA_PATH, INDICES_PATH
from rag_it.utils import _check_if_index_exists
import os


class Engine:
    def __init__(self, llm, embed_model, index, data_path) -> None:
        self.llm = llm
        self.embed_model = embed_model
        self._settings = Settings
        self._init_settings()
        self.index = self._load_index(index=index, data_directory=data_path)

    def _init_settings(self):
        self._settings.llm = self.llm.model
        self._settings.embed_model = self.embed_model.model

    def _load_storage_context(self, index):
        chroma_client = chromadb.PersistentClient(path=INDICES_PATH)
        chroma_collection = chroma_client.get_or_create_collection(index)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return storage_context

    def _load_data(self, data_directory):
        documents = SimpleDirectoryReader(os.path.join(DATA_PATH, data_directory)).load_data()
        return documents

    def _load_index(self, index, data_directory):
        documents = self._load_data(data_directory=data_directory)
        storage_context = self._load_storage_context(index=index)
        search_index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)
        return search_index

    def query(self, input):
        query_engine = self.index.as_query_engine()
        response = query_engine.query(input)
        return response
