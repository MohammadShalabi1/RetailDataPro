from app.core.config import Settings, get_settings


class HealthRepository:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def is_database_configured(self) -> bool:
        return self._settings.database_configured


def get_health_repository() -> HealthRepository:
    return HealthRepository()
