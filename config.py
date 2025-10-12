"""
Configuration module for ResuMatch AI
Provides a Config class and a config mapping compatible with existing imports.
"""

import os
from pathlib import Path


class Config:
    """Base configuration.
    Access via config['default'] or import Config directly.
    """

    # General
    DEBUG: bool = os.getenv("DEBUG", "0").lower() in {"1", "true", "yes", "on"}
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_FILE: str = os.getenv("LOG_FILE", str(Path("logs") / "app.log"))

    # Models
    SENTENCE_TRANSFORMER_MODEL: str = os.getenv(
        "SENTENCE_TRANSFORMER_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    @staticmethod
    def get_openai_api_key() -> str:
        """Return the OpenAI API key from environment variables."""
        return os.getenv("OPENAI_API_KEY")


class DevelopmentConfig(Config):
    DEBUG: bool = True
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "DEBUG").upper()
    LOG_FILE: str = os.getenv("LOG_FILE", str(Path("logs") / "app-dev.log"))


class TestingConfig(Config):
    DEBUG: bool = False
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "WARNING").upper()
    LOG_FILE: str = os.getenv("LOG_FILE", str(Path("logs") / "app-test.log"))


class ProductionConfig(Config):
    DEBUG: bool = os.getenv("DEBUG", "0").lower() in {"1", "true", "yes", "on"}
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_FILE: str = os.getenv("LOG_FILE", str(Path("logs") / "app.log"))


# Mapping used by existing code: config['default'] ...
config = {
    "default": Config,
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}
