"""StoreOpsAgent - FastAPI application entry point."""

from fastapi import FastAPI

from app.api import question_router

app = FastAPI(
    title="StoreOpsAgent",
    description="AI-powered store operations assistant",
    version="0.1.0",
)

# Include routers
app.include_router(question_router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to StoreOpsAgent API"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
