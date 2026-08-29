from functools import lru_cache
from decimal import Decimal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "RetailData-Pro API"
    app_env: str = "development"
    api_cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    database_url: str = "postgresql+psycopg://retaildata_user:change-me@localhost:5432/retaildata_pro"
    readonly_database_url: str | None = None
    gemini_api_key: str | None = None
    gemini_text_model: str = "gemini-3.5-flash-lite"
    gemini_structured_model: str = "gemini-3.6-flash"
    gemini_embedding_model: str = "text-embedding-004"
    embedding_dimension: int = 768
    ai_provider_timeout_seconds: float = 30.0
    ai_provider_max_retries: int = 2
    rag_dense_top_k: int = 20
    rag_lexical_top_k: int = 20
    rag_rrf_k: int = 60
    rag_rerank_top_k: int = 8
    rag_context_top_k: int = 6
    gemini_35_flash_lite_input_cost_per_1m: Decimal = Decimal("0.30")
    gemini_35_flash_lite_output_cost_per_1m: Decimal = Decimal("2.50")
    gemini_36_flash_input_cost_per_1m: Decimal = Decimal("1.50")
    gemini_36_flash_output_cost_per_1m: Decimal = Decimal("7.50")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("api_cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("database_url", "readonly_database_url", mode="before")
    @classmethod
    def use_psycopg_driver(cls, value: str | None) -> str | None:
        if isinstance(value, str) and value.startswith("postgresql://"):
            return value.replace("postgresql://", "postgresql+psycopg://", 1)
        return value

    @property
    def is_development(self) -> bool:
        return self.app_env.lower() == "development"

    @property
    def database_configured(self) -> bool:
        return self.database_url.startswith("postgresql")

    @property
    def readonly_database_configured(self) -> bool:
        return self.readonly_database_url is not None and self.readonly_database_url.startswith("postgresql")


@lru_cache
def get_settings() -> Settings:
    return Settings()
