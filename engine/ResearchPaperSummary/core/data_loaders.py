from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from core.singleton import Singleton

import logging


class PDFLoader(Singleton):
    def __init__(self, pdf_path: str):
        self.__loader = PyMuPDFLoader(pdf_path)
        self.__docs: list[Document] | None = None

    @property
    def docs(self) -> list[Document]:
        logging.info(f"Pages : {len(self.__docs)}")
        return self.__docs

    def load(self) -> None:
        try:
            self.__docs = self.__loader.load()

        except Exception as e:
            logging.error(f"Error loading PDF: {e}")
            raise
