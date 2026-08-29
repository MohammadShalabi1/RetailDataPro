from fastapi import Depends

from app.repositories.health_repository import HealthRepository, get_health_repository
from app.schemas.health import HealthResponse


class HealthService:
    def __init__(self, health_repository: HealthRepository) -> None:
        self._health_repository = health_repository

    def get_status(self) -> HealthResponse:
        return HealthResponse(
            status="ok",
            service="retaildata-pro-api",
            database_configured=self._health_repository.is_database_configured(),
        )


def get_health_service(
    health_repository: HealthRepository = Depends(get_health_repository),
) -> HealthService:
    return HealthService(health_repository)
