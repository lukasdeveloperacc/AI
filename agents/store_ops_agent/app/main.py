"""StoreOpsAgent - FastAPI application entry point."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat_router, question_router
from app.middleware import ErrorHandlerMiddleware, LatencyMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

app = FastAPI(
    title="StoreOpsAgent",
    description="""
AI-powered store operations assistant with RAG capabilities.

## Features
- **Chat Endpoint** (`POST /api/v1/chat`): Ask questions and get answers with citations
- **Question Endpoint** (`POST /api/v1/ask`): Alternative question answering endpoint
- **Health Check** (`GET /health`): System health monitoring

## Error Handling
All errors return a consistent JSON format with:
- `error.code`: Machine-readable error code
- `error.message`: Human-readable error message
- `error.trace_id`: Unique ID for debugging
- `error.details`: Additional context (optional)

## Response Headers
- `X-Trace-ID`: Unique request identifier for debugging
- `X-Response-Time-Ms`: Request processing time in milliseconds
""",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    contact={
        "name": "StoreOps Team",
    },
    license_info={
        "name": "MIT",
    },
)

# Add middleware (order matters: first added = last executed)
# Error handler should be outermost to catch all errors
app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(LatencyMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router)
app.include_router(question_router)


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to StoreOpsAgent API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint for monitoring and load balancers."""
    return {"status": "healthy", "service": "store-ops-agent"}
