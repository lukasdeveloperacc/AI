from langchain_core.runnables import Runnable, RunnableLambda, RunnableConfig
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

from chains.preprocess_models import DocumentPreprocessInput, DocumentPreprocessOutput

from typing import Any

import base64, tempfile, io, logging


class DocumentPreprocessor(Runnable[DocumentPreprocessInput, DocumentPreprocessOutput]):
    def __init__(
        self,
        loader_cls: type[PyMuPDFLoader],
        splitter: RecursiveCharacterTextSplitter,
        loader_args: dict[str, Any] = {},
    ):
        super().__init__()

        self._loader_cls = loader_cls
        self._loader_args = loader_args
        self._splitter = splitter
        self._chain = (
            RunnableLambda(DocumentPreprocessInput.from_output)
            | RunnableLambda(self.route)
            | RunnableLambda(self.load_pdf)
            | RunnableLambda(self.split_docs)
        )

    @staticmethod
    def load_pdf_from_base64(_input: DocumentPreprocessInput) -> DocumentPreprocessInput:
        try:
            pdf_bytes: io.BytesIO = base64.b64decode(_input.base64_file)

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                logging.info(f"Save from base64 string to pdf file : {tmp.name}")

                tmp.write(pdf_bytes)
                tmp.flush()

                _input.pdf_path = tmp.name

            return _input

        except Exception as e:
            logging.error(e)
            raise

    def route(self, _input: DocumentPreprocessInput):
        try:
            if _input.base64_file:
                return RunnableLambda(self.load_pdf_from_base64)
            else:
                return _input

        except Exception as e:
            logging.error(e)
            raise

    def load_pdf(self, _input: DocumentPreprocessInput) -> list[Document]:
        try:
            loader = self._loader_cls(_input.pdf_path, **self._loader_args)
            return loader.load()

        except Exception as e:
            logging.error(e)
            raise

    def split_docs(self, _input: list[Document]) -> list[Document]:
        try:
            return self._splitter.split_documents(_input)

        except Exception as e:
            logging.error(e)
            raise

    def invoke(self, _input: DocumentPreprocessInput, config: RunnableConfig | None = None) -> DocumentPreprocessOutput:
        logging.info(f"{__class__.__name__} Invoke")
        docs = self._chain.invoke(_input, config)

        return DocumentPreprocessOutput(docs=docs)
