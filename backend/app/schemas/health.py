from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    database_configured: bool
