from langchain_community.vectorstores import FAISS
from custom_types.vector_store import VectorStoreType
from core.singleton import Singleton
from typing import Optional

import logging


class VectorStore(Singleton):
    def __init__(self, db_type: str = VectorStoreType.FAISS.value, *args, **kwargs):
        self._db = None

        if db_type == VectorStoreType.FAISS.value:
            self._db = FAISS.from_documents(*args, **kwargs)
        else:
            logging.warning(f"Unknown db type : {db_type}")

    @property
    def db(self) -> Optional[FAISS]:
        return self._db
