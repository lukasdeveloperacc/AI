from core.data_loaders import PDFLoader
from core.splitters import DocsSplitter
from langchain.schema.document import Document
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI
from typing import Dict, Any, List, AsyncIterator, Iterator
from schemas.types import FileProcessingRequest

import logging, os, base64, uuid


class ResearchPaperSummaryChain(Runnable):
    def __init__(self):
        super().__init__()
        self.__docs: list[Document] = []

        self.__map_llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
        self.__reduce_llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

    @property
    def chain(self) -> LLMChain:
        try:

            def _preprocess_wrapper(input: FileProcessingRequest) -> Dict[str, Any]:
                if isinstance(input, FileProcessingRequest):
                    return {
                        "splited_docs": self.__perprocess(input.base64_file),
                        "language": input.language,
                    }
                else:
                    raise TypeError(f"Unkown tpye of input : {type(input)}")

            def _map_chain_wrapper(input: Dict[str, Any]) -> Dict[str, Any]:
                splited_docs = input.get("splited_docs")
                if not isinstance(splited_docs, list):
                    raise ValueError("splited_docs should be a list of Documents")

                doc_texts = [doc.page_content for doc in splited_docs]

                return {
                    "doc_summaries": self.map_chain.batch([{"doc": doc} for doc in doc_texts]),
                    "language": input.get("language"),
                }

            return RunnableLambda(_preprocess_wrapper) | RunnableLambda(_map_chain_wrapper) | self.reduce_chain

        except Exception as e:
            raise Exception(f"Fail to make chain : {e}")

    @property
    def map_chain(self) -> LLMChain:
        return self.__make_map_prompt() | self.__map_llm | StrOutputParser()

    @property
    def reduce_chain(self) -> LLMChain:
        return self.__make_reduce_prompt() | self.__reduce_llm | StrOutputParser()

    def __perprocess(self, base64_str: str) -> list[Document]:
        try:
            pdf_file_path = f"/data/{uuid.uuid4()}.pdf"
            if base64_str is not None:
                logging.info(f"Trying to write to {pdf_file_path}")
                bytes_content: bytes = base64.b64decode(base64_str.encode("utf-8"))
                os.makedirs(os.path.dirname(pdf_file_path), exist_ok=True)
                with open(pdf_file_path, "wb") as f:
                    f.write(bytes_content)
                logging.info(f"Success to write to {pdf_file_path}")

            # Load PDF for getting documents
            data_loader = PDFLoader.instance(pdf_file_path)
            data_loader.load()

            # Split documents
            self.__docs = data_loader.docs
            splitter = DocsSplitter.instance(chunk_size=1000, chunk_overlap=50)
            splitter.split(self.__docs)

            return splitter.splitted_docs

        except Exception as e:
            raise Exception(f"Fail to preprocess : {e}")

    def make_prompt(self, prompt: str | None = None):
        try:
            return PromptTemplate.from_template(prompt)

        except Exception as e:
            raise Exception(f"Fail to make prompt : {e}")

    def __make_map_prompt(self, prompt: str | None = None) -> PromptTemplate:
        if prompt is None:
            prompt: str = """
            ================================ System Message ================================
                            You are a professional main thesis extractor.
            ================================ Human Message =================================

            Your task is to extract main thesis from given documents. Answer should be in same language as given document. 

            #Format: 
            - thesis 1
            - thesis 2
            - thesis 3
            - ...

            Here is a given document: 
            {doc}

            Write 1~5 sentences.
            #Answer:"""

        return self.make_prompt(prompt)

    def __make_reduce_prompt(self, prompt: str | None = None) -> PromptTemplate:
        if prompt is None:
            prompt: str = """================================ System Message ================================

                            You are a professional summarizer. You are given a list of summaries of documents and you are asked to create a single summary of the documents.

                            ================================ Human Message =================================

                            #Instructions: 
                            1. Extract main points from a list of summaries of documents
                            2. Make final summaries in bullet points format.
                            3. Answer should be written in {language}.

                            #Format: 
                            - summary 1
                            - summary 2
                            - summary 3
                            - ...

                            Here is a list of summaries of documents: 
                            {doc_summaries}

                            #SUMMARY:"""

            return self.make_prompt(prompt)

    def invoke(self, input: Dict[str, Any], config: Any = None) -> str:
        try:
            logging.info("Invoking chain")
            result = self.chain.invoke(input, config=config)
            logging.info("Invoke success")

            return result

        except Exception as e:
            logging.error(e)

            return f"Server Error : {e}"

    async def ainvoke(self, input: Dict[str, Any], config: Any = None) -> str:
        try:
            logging.info("Ainvoking chain")
            result = await self.chain.ainvoke(input, config=config)
            logging.info("Ainvoking Success")

            return result

        except Exception as e:
            logging.error(e)

            return f"Server Error : {e}"

    def stream(self, input: Dict[str, Any], config: Any = None) -> Iterator[str]:
        try:
            logging.info("Streaming chain")
            for chunk in self.chain.stream(input, config=config):
                yield chunk

        except Exception as e:
            logging.error(f"Streaming Error : {e}")

    async def astream(self, input: Dict[str, Any], config: Any = None) -> AsyncIterator[str]:
        try:
            logging.info("Astreaming chain")
            async for chunk in self.chain.astream(input, config=config):
                yield chunk

        except Exception as e:
            logging.error(f"Astreaming Error : {e}")

    def batch(self, inputs: List[Dict[str, Any]], config: Any = None) -> List[str]:
        return self.chain.batch(inputs, config=config)
