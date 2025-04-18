from langchain.text_splitter import RecursiveCharacterTextSplitter
from core.singleton import Singleton
from langchain_core.documents import Document
import logging


class DocsSplitter(Singleton):
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.__splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.__splitted_docs = None

    @property
    def splitted_docs(self) -> list[Document]:
        return self.__splitted_docs

    def split(self, docs: list[Document]) -> None:
        try:
            self.__splitted_docs = self.__splitter.split_documents(docs)

        except Exception as e:
            logging.error(f"Error splitting documents: {e}")
            raise
