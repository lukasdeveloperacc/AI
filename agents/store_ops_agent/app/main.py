"""StoreOpsAgent - FastAPI application entry point."""

from fastapi import FastAPI

app = FastAPI(
    title="StoreOpsAgent",
    description="AI-powered store operations assistant",
    version="0.1.0",
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to StoreOpsAgent API"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
