from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

# from chains.task_chains.summary_chains import ResearchPaperSummaryChain
# from chains.task_chains.summary_models import SummaryChainInput

import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(api_key="")
prompt = PromptTemplate.from_template("{topic} 에 대하여 3문장으로 설명해줘.")
chain = prompt | model | StrOutputParser()
# summary_chain = ResearchPaperSummaryChain(preprocess_args={"chunk_size": 1000, "chunk_overlap": 50})

add_routes(app, chain)

if __name__ == "__main__":
    import uvicorn, os

    uvicorn.run(
        "server:app",
        host=os.getenv("HOST", os.getenv("HOST", "127.0.0.1")),
        port=os.getenv("PORT", 8000),
        reload=True,
    )
