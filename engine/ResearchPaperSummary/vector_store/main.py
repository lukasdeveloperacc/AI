from langchain_community.vectorstores import FAISS

import os, logging

os.environ["FAISS_NO_GPU"] = "1"


class DB:
    db: FAISS | None = None

    def store(self, documents: list[str], embeddings) -> None:
        try:
            logging.info(f"Trying to store")
            self.db = FAISS.from_documents(documents, embeddings)
            logging.info(f"Success to store")

        except Exception as e:
            raise Exception(f"Fail to store : {e}")
