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
    VERSION: str = "0.1.0"
    DEBUG: bool = False
    
    # Paths
    DATA_DIR: Path = Path("data")
    WALLS_DIR: Path = DATA_DIR / "walls"
    MODELS_DIR: Path = DATA_DIR / "models"
    DB_PATH: Path = DATA_DIR / "storage.db"
    
    # Pagination defaults
    LIMIT: int = 50
    
    # Model defaults
    NUM_LIMBS: int = 2
    NUM_FEATURES: int = 5
    N_HIDDEN_LAYERS: int = 3
    NULL_FEATURES: list[float] = [-1.0,-1.0,0,0,-1.0]
    HIDDEN_DIM: int = 256

    # Training defaults
    AUGMENT_DATA: bool = True
    VAL_SPLIT: float = 0.2
    EPOCHS: int = 100
    MAX_EPOCHS: int = 1000
    LR: float = 0.001
    DEVICE: str = "cpu"

    # Test settings
    TEST_ASSETS_DIR: Path = Path("test_assets")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience access
settings = get_settings()
