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
    ClimbBatchCreate,
    ClimbBatchCreateResult,
    ClimbBatchCreateResponse,
    ClimbSortBy,
    ClimbListResponse,
    ClimbCreateResponse,
    ClimbDeleteResponse,
)
from app.schemas.generate import (
    GradeScale,
    GenerateRequest,
    GenerateResponse,
)

__all__ = [
    # Base
    "PositiveInt",
    "HoldPosition",
    "HoldDetail",
    "Holdset",
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
    # Generation
    "GradeScale",
    "GenerateRequest",
    "GenerateResponse",
]
