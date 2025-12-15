"""API routes package."""

from app.api.chat import router as chat_router
from app.api.question import router as question_router

__all__ = ["chat_router", "question_router"]
