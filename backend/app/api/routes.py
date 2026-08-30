from fastapi import APIRouter

from app.api.v1.analytics import router as analytics_router
from app.api.v1.chat import router as chat_router
from app.api.v1.conversations import router as conversations_router
from app.api.v1.documents import router as documents_router
from app.api.v1.evaluations import router as evaluations_router
from app.api.v1.health import router as health_router
from app.api.v1.insights import router as insights_router
from app.api.v1.observability import router as observability_router
from app.api.v1.reports import router as reports_router

api_router = APIRouter()
api_router.include_router(analytics_router)
api_router.include_router(chat_router)
api_router.include_router(conversations_router)
api_router.include_router(documents_router)
api_router.include_router(evaluations_router)
api_router.include_router(health_router)
api_router.include_router(insights_router)
api_router.include_router(observability_router)
api_router.include_router(reports_router)
