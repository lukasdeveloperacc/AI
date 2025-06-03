from langchain_core.runnables import Runnable, RunnableConfig
from langchain.document_loaders.base import BaseLoader
from langchain_text_splitters.base import TextSplitter
from langchain_core.retrievers import BaseRetriever
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from typing import Type, TypeAlias, Union

from chains.retriever_io import RetrieverInput

import logging
import os

os.environ["FAISS_NO_GPU"] = "1"

load_dotenv()

MyVectorStoreType: TypeAlias = Union[VectorStore, FAISS]


class BaseRetrieverChain(Runnable):
    def __init__(
        self,
        vector_store,
        vector_store_path=None,
        documents=None,
        loader=None,
        splitter=None,
        embdding=None,
    ):
        super().__init__()
        self._documents = documents
        self._loader: BaseLoader = loader
        self._splitter: TextSplitter = splitter
        self._embedding: Embeddings | None = embdding
        self._vector_store: Type[VectorStore] | MyVectorStoreType = vector_store
        self._retriever: BaseRetriever | VectorStoreRetriever = self.make_retriever(vector_store_path=vector_store_path)

    @property
    def retriever(self) -> BaseRetriever | VectorStoreRetriever:
        return self._retriever

    @property
    def db(self) -> MyVectorStoreType:
        return self._vector_store

    def make_retriever(self, **kwargs) -> BaseRetriever | VectorStoreRetriever:
        docs = self._loader.load() if self._loader else self._documents
        splitted_docs = self._splitter.split_documents(docs) if self._splitter else docs
        logging.info(f"Doc like : {splitted_docs[0]}")

        vector_store_path = kwargs.get("vector_store_path")
        if not self._embedding:
            raise ValueError(f"Need embedding parameters : {self._embedding}")

        if vector_store_path and os.path.exists(vector_store_path):
            logging.info(f"Load from {vector_store_path}")
            self._vector_store = self._vector_store.load_local(
                vector_store_path, self._embedding, allow_dangerous_deserialization=True
            )
        elif isinstance(self._vector_store, type):
            self._vector_store: MyVectorStoreType = self._vector_store.from_documents(
                documents=splitted_docs, embedding=self._embedding
            )

            if vector_store_path:
                os.makedirs(vector_store_path, exist_ok=True)
                logging.info(f"Save in {vector_store_path}")
                self._vector_store.save_local(vector_store_path)

        return self._vector_store.as_retriever(**kwargs)

    def invoke(self, input: RetrieverInput, config: RunnableConfig | None = None) -> str:
        return self._retriever.invoke(input, config)
