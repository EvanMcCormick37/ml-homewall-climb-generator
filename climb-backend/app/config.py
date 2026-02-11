"""
Application configuration.

Uses pydantic-settings to load from environment variables with sensible defaults.
"""
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # App metadata
    NAME: str = "Beta Zero API"
    VERSION: str = "0.2.0"
    DEBUG: bool = False
    
    # Paths
    DATA_DIR: Path = Path("data")
    WALLS_DIR: Path = DATA_DIR / "walls"
    DB_PATH: Path = DATA_DIR / "storage.db"
    
    # DDPM model paths & hyperparams
    DDPM_WEIGHTS_PATH: Path = DATA_DIR / "models" / "ddpm-weights.pth"
    SCALER_WEIGHTS_PATH: Path = DATA_DIR / "models" / "scaler-weights.joblib"
    HC_WEIGHTS_PATH: Path = DATA_DIR / "models" / "unet-hold-classifier.pth"
    
    # Pagination defaults
    LIMIT: int = 50

    # Test settings
    TEST_ASSETS_DIR: Path = Path("test_assets")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience access
settings = get_settings()
