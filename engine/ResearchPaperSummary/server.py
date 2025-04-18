from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

from chain.main import ResearchPaperSummaryChain
from chain.chat import ResearchPaperChatChain
from schemas.types import FileProcessingRequest, ChatReuqest

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

summary_chain = ResearchPaperSummaryChain().with_types(input_type=FileProcessingRequest)
chat_chain = ResearchPaperChatChain().with_types(input_type=ChatReuqest)

add_routes(app, summary_chain, path="/summary")
add_routes(app, chat_chain, path="/chat")

if __name__ == "__main__":
    import uvicorn, os

    uvicorn.run(
        "server:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=os.getenv("PORT", 8000),
        reload=True,
    )
