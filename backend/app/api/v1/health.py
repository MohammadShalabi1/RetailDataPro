from fastapi import APIRouter, Depends

from app.schemas.health import HealthResponse
from app.services.health_service import HealthService, get_health_service

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def get_health(
    health_service: HealthService = Depends(get_health_service),
) -> HealthResponse:
    return health_service.get_status()
