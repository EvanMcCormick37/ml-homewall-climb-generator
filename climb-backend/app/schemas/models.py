"""
Pydantic schemas for ML model-related requests and responses.
"""
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
from app.schemas.base import PositiveInt, HoldPosition

class ModelType(str, Enum):
    """Available model architectures."""
    MLP = "mlp"
    RNN = "rnn"
    LSTM = "lstm"


class ModelStatus(str, Enum):
    """Model training status."""
    UNTRAINED = "untrained"
    TRAINING = "training"
    TRAINED = "trained"
    FAILED = "failed"


# --- Feature Configuration ---

class FeatureConfig(BaseModel):
    """Configuration for which features to use in training/generation."""
    position: bool = True
    pull_direction: bool = True
    difficulty: bool = True


# --- Model Schemas ---

class ModelCreate(BaseModel):
    """Schema for creating and training a model."""
    model_type: ModelType = ModelType.MLP
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    epochs: int = Field(100, ge=1, le=10000)
    augment_dataset: bool = True
    val_split: float = 0.2


class ModelSummary(BaseModel):
    """Detailed model info."""
    id: str
    wall_id: str
    model_type: ModelType
    features: FeatureConfig
    status: ModelStatus
    val_loss: float | None = None
    epochs_trained: int
    climbs_trained: int
    moves_trained: int
    created_at: datetime
    trained_at: datetime | None = None


class ModelCreateResponse(BaseModel):
    """Response after creating a model (training starts)."""
    model_id: str
    job_id: str


class ModelListResponse(BaseModel):
    """Response for listing models."""
    models: list[ModelSummary]
    total: int


class ModelDeleteResponse(BaseModel):
    """Response after deleting a model."""
    id: str


# --- Generation Schemas ---

class GenerateRequest(BaseModel):
    """Request schema for generating climbs."""
    starting_holds: HoldPosition
    stop_holds: list[PositiveInt] = Field(
        default_factory=list,
        description="Optional List of 'stop holds' on which to force stop the climb"
    )
    max_moves: int = Field(10, ge=1, le=50)
    num_climbs: int = Field(5, ge=1, le=20)
    temperature: float = Field(0.01, ge=0.001, le=1.0)
    force_alternating: bool = Field(
        True,
        description="Force one limb to move at a time"
    )


class GeneratedClimb(BaseModel):
    """A single generated climb."""
    sequence: list[HoldPosition]
    num_moves: int


class GenerateResponse(BaseModel):
    """Response containing generated climbs."""
    model_id: str
    climbs: list[GeneratedClimb]
    num_generated: int
    parameters: GenerateRequest
