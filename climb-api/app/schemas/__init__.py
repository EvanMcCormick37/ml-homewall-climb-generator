"""
Pydantic schemas for request/response validation.
"""
from app.schemas.walls import (
    Hold,
    HoldCreate,
    WallCreate,
    WallDetail,
    WallMetadata,
    WallListResponse,
    WallCreateResponse,
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
    ModelDetail,
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
    # Walls
    "Hold",
    "HoldCreate", 
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
    "ModelDetail",
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
