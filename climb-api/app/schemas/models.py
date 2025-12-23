"""
Pydantic schemas for ML model-related requests and responses.
"""
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
from typing import Annotated

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
    epochs: int = Field(100, ge=1, le=1000)
    augment_dataset: bool = True


class ModelSummary(BaseModel):
    """Brief model info for listing."""
    id: str
    model_type: ModelType
    status: ModelStatus
    moves_trained: float
    climbs_trained: float
    val_loss: float | None
    created_at: datetime
    trained_at: datetime | None


class ModelDetail(BaseModel):
    """Detailed model info."""
    id: str
    wall_id: str
    model_type: ModelType
    features: FeatureConfig
    status: ModelStatus
    val_loss: float | None
    epochs_trained: int
    created_at: datetime
    trained_at: datetime | None


class ModelCreateResponse(BaseModel):
    """Response after creating a model (training starts)."""
    model_id: str
    job_id: str
    message: str = "Model training started"


class ModelListResponse(BaseModel):
    """Response for listing models."""
    models: list[ModelSummary]
    total: int


class ModelDeleteResponse(BaseModel):
    """Response after deleting a model."""
    id: str
    message: str = "Model deleted successfully"


# --- Generation Schemas ---
PositiveInt = Annotated[int,Field(ge=0)]

class GenerateRequest(BaseModel):
    """Request schema for generating climbs."""
    starting_holds: list[PositiveInt] = Field(
        ..., 
        min_length=2, 
        max_length=2,
        description="[left_hand_hold_id, right_hand_hold_id]"
    )
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
    sequence: list[list[int]]
    num_moves: int


class GenerateResponse(BaseModel):
    """Response containing generated climbs."""
    model_id: str
    climbs: list[GeneratedClimb]
    num_generated: int
    parameters: GenerateRequest
