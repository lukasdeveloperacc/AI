from langchain.text_splitter import RecursiveCharacterTextSplitter

import logging


class DocsSpliter:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.__splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.__splitted_docs = None

    def split(self, docs: list[str]) -> None:
        try:
            self.__splitted_docs = self.__splitter.split_documents(docs)

        except Exception as e:
            raise Exception(f"Fail to split docs : {e}")

    @property
    def splitted_docs(self) -> list[str]:
        logging.info(f"Get splitted docs")
        return self.__splitted_docs
