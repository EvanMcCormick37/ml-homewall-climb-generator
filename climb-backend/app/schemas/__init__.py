"""
Pydantic schemas for request/response validation.
"""
from app.schemas.base import (
    PositiveInt,
    HoldPosition,
    HoldDetail,
    Holdset,
)
from app.schemas.walls import (
    WallCreate,
    WallDetail,
    WallMetadata,
    WallListResponse,
    WallCreateResponse,
    SetHoldsResponse,
)
from app.schemas.climbs import (
    Climb,
    ClimbCreate,
    ClimbSortBy,
    ClimbListResponse,
    ClimbCreateResponse,
    ClimbDeleteResponse,
)
from app.schemas.models import (
    ModelType,
    ModelStatus,
    FeatureConfig,
    ModelCreate,
    ModelSummary,
    ModelCreateResponse,
    ModelListResponse,
    ModelDeleteResponse,
    GenerateRequest,
    GeneratedClimb,
    GenerateResponse,
)
from app.schemas.jobs import (
    Job,
    JobStatus,
    JobType,
)

__all__ = [
    # Base
    "PositiveInt",
    "HoldPosition",
    "HoldDetail",
    # Walls
    "WallCreate",
    "WallDetail",
    "WallMetadata",
    "WallListResponse",
    "WallCreateResponse",
    # Climbs
    "Climb",
    "ClimbCreate",
    "ClimbSortBy",
    "ClimbListResponse",
    "ClimbCreateResponse",
    "ClimbDeleteResponse",
    # Models
    "ModelType",
    "ModelStatus",
    "FeatureConfig",
    "ModelCreate",
    "ModelSummary",
    "ModelCreateResponse",
    "ModelListResponse",
    "ModelDeleteResponse",
    "GenerateRequest",
    "GeneratedClimb",
    "GenerateResponse",
    # Jobs
    "Job",
    "JobStatus",
    "JobType",
]
