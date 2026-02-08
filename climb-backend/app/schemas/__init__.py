"""
Pydantic schemas for request/response validation.
"""
from app.schemas.base import (
    PositiveInt,
    HoldPosition,
    HoldDetail,
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
    ClimbBatchCreate,
    ClimbBatchCreateResult,
    ClimbBatchCreateResponse,
    ClimbSortBy,
    ClimbListResponse,
    ClimbCreateResponse,
    ClimbDeleteResponse,
    Holdset,
)
from app.schemas.generate import (
    GradeScale,
    GenerateRequest,
    GeneratedClimb,
    GenerateResponse,
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
    "SetHoldsResponse",
    # Climbs
    "Climb",
    "ClimbCreate",
    "ClimbBatchCreate",
    "ClimbBatchCreateResult",
    "ClimbBatchCreateResponse",
    "ClimbSortBy",
    "ClimbListResponse",
    "ClimbCreateResponse",
    "ClimbDeleteResponse",
    "Holdset",
    # Generation
    "GradeScale",
    "GenerateRequest",
    "GeneratedClimb",
    "GenerateResponse",
]
