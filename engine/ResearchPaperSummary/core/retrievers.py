from langchain.schema.vectorstore import VectorStoreRetriever
from langchain_core.vectorstores import VectorStore
from core.singleton import Singleton
from typing import Optional


class Retriever(Singleton):
    def __init__(self, db: VectorStore, *args, **kwargs):
        self._retriever = db.as_retriever(*args, **kwargs)

    @property
    def retriever(self) -> Optional[VectorStoreRetriever]:
        return self._retriever
