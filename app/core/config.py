
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Central application settings loaded from environment variables."""

    app_name: str = "SciFact Research Assistant"
    environment: str = "development"
    log_level: str = "INFO"

    gemini_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="GEMINI_API_KEY",
    )

    max_upload_bytes: int = Field(
        default=15 * 1024 * 1024,
        gt=0,
    )

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Load settings once and reuse them across the application."""
    return Settings()
