from chain.main import ResearchPaperSummaryChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.vectorstore import VectorStoreRetriever
from typing import AsyncIterator, Iterator, Dict, Any
from schemas.types import ChatReuqest
from core.retrievers import Retriever
from core.vector_store import VectorStore
from core.embeddings import Embedding
from core.splitters import DocsSplitter

import logging


class ResearchPaperChatChain(ResearchPaperSummaryChain):
    def __init__(self):
        super().__init__()
        self.__chat_prompt: PromptTemplate = self._make_prompt()
        self.__chat_llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

    @property
    def chain(self) -> LLMChain:
        try:

            def _preprocess_wrapper(input: ChatReuqest) -> Dict[str, Any]:
                self.__perprocess()
                docs = Retriever.instance().retriever.get_relevant_documents(input.question)
                context = "\n\n".join([doc.page_content for doc in docs])
                return {
                    "context": context,
                    "question": input.question,
                }

            return RunnableLambda(_preprocess_wrapper) | self.__chat_prompt | self.__chat_llm | StrOutputParser()

        except Exception as e:
            raise Exception(f"Fail to make chain : {e}")

    def __perprocess(self) -> None:
        try:
            documents = DocsSplitter.instance().splitted_docs
            if documents is None:
                raise Exception("No documents to split")
            else:
                logging.info(f"Splitted documents : {len(documents) if documents else type(documents)}")

            embedding = Embedding.instance()
            vector_store = VectorStore.instance(documents=documents, embedding=embedding.embedding)
            _ = Retriever.instance(db=vector_store.db)

        except Exception as e:
            raise Exception(f"Fail to preprocess : {e}")

    def _make_prompt(self) -> PromptTemplate:
        prompt = PromptTemplate.from_template(
            """You are an assistant for question-answering tasks. 
                Use the following pieces of retrieved context to answer the question. 
                If you don't know the answer, just say that you don't know. 
                Answer in Korean.

                #Question: 
                {question} 
                #Context: 
                {context} 

                #Answer:"""
        )

        return prompt

    def invoke(self, input, config=None) -> str:
        logging.info(f"Input : {input}")
        return self.chain.invoke(input, config)

    def ainvoke(self, input, config=None) -> str:
        return self.chain.ainvoke(input, config)

    def astream(self, input, config=None) -> AsyncIterator:
        return self.chain.astream(input, config)

    def stream(self, input, config=None) -> Iterator:
        return self.chain.stream(input, config)

    def batch(self, inputs, config=None) -> str:
        return self.chain.batch(inputs, config)
