from langchain_openai import OpenAIEmbeddings
from custom_types.embedding import EmbeddingType
from core.singleton import Singleton
from typing import Optional

import logging


class Embedding(Singleton):
    def __init__(self, embedding_type: str = EmbeddingType.OPENAI.value, *args, **kwargs):
        self._embedding = None

        if embedding_type == EmbeddingType.OPENAI.value:
            self._embedding = OpenAIEmbeddings(*args, **kwargs)
        else:
            logging.warning(f"Unknown embedding type : {embedding_type}")

    @property
    def embedding(self) -> Optional[OpenAIEmbeddings]:
        return self._embedding
