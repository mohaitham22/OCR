"""Environment-backed settings.

Every default here must be safe on a machine with no API key, no database and
no OCR engine installed, because import of this module happens before anything
else in the app.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Extraction backend ---------------------------------------------
    default_engine: str = Field(
        default="traditional",
        description="Engine used when a request does not name one: traditional | vlm.",
    )

    # --- Model access ----------------------------------------------------
    anthropic_api_key: str = Field(default="", description="Empty disables both LLM paths.")
    llm_model: str = Field(default="claude-opus-5", description="Text model for app.llm.structured_text.")
    vision_model: str = Field(default="claude-opus-5", description="Vision model for app.llm.structured_vision.")
    llm_max_tokens: int = 8192
    llm_timeout_seconds: float = 120.0
    llm_max_retries: int = Field(default=3, description="Retries on a schema-invalid response.")

    # --- OCR -------------------------------------------------------------
    ocr_backend: str = Field(default="paddle", description="paddle | tesseract.")
    ocr_languages: str = Field(
        default="ar,en",
        description="Comma-separated language codes in the backend's own vocabulary.",
    )
    ocr_min_confidence: float = 0.50
    tesseract_cmd: str | None = Field(
        default=None,
        description="Absolute path to tesseract.exe when it is not on PATH.",
    )

    # --- Preprocessing ---------------------------------------------------
    pdf_dpi: int = Field(default=300, description="Rasterisation DPI for image-only PDF pages.")
    max_image_px: int = Field(default=2500, description="Longest edge after downscaling.")
    deskew: bool = True
    auto_crop: bool = True
    fix_lighting: bool = True

    # --- Validation and review gate --------------------------------------
    amount_tolerance: float = Field(
        default=0.02,
        description="Absolute currency slack when checking that lines sum to the total.",
    )
    review_confidence_threshold: float = Field(
        default=0.80,
        description="Extractions below this confidence go to the review queue.",
    )

    # --- Persistence (optional) ------------------------------------------
    database_url: str = Field(
        default="",
        description="Empty means run without Postgres; only storage is skipped.",
    )
    db_pool_min: int = 1
    db_pool_max: int = 5

    # --- Runtime ---------------------------------------------------------
    log_level: str = "INFO"
    max_upload_mb: int = 20
    data_dir: Path = Path("data")

    @field_validator("default_engine", "ocr_backend", "log_level", mode="before")
    @classmethod
    def _normalise(cls, v: str) -> str:
        return v.strip().lower() if isinstance(v, str) else v

    @property
    def languages(self) -> list[str]:
        return [part.strip().lower() for part in self.ocr_languages.split(",") if part.strip()]

    @property
    def persistence_enabled(self) -> bool:
        return bool(self.database_url.strip())

    @property
    def llm_enabled(self) -> bool:
        return bool(self.anthropic_api_key.strip())


settings = Settings()
