"""
Pydantic schemas for request/response validation.
"""
from app.schemas.base import (
    PositiveInt,
    HoldPosition,
    HoldDetail,
    Holdset,
)

from app.schemas.sizes import (
    SizeMetadata,
    SizeCreate,
    SizeCreateResponse,
)
from app.schemas.layouts import (
    LayoutCreate,
    LayoutEdit,
    LayoutDetail,
    LayoutMetadata,
    LayoutListResponse,
    LayoutCreateResponse,
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
    GenerateSettings,
    GenerateResponse,
)

__all__ = [
    # Base
    "PositiveInt",
    "HoldPosition",
    "HoldDetail",
    "Holdset",
    # Layouts (new)
    "LayoutCreate",
    "LayoutEdit",
    "LayoutDetail",
    "LayoutMetadata",
    "LayoutListResponse",
    "LayoutCreateResponse",
    "SetHoldsResponse",
    # Sizes (new)
    "SizeMetadata",
    "SizeCreate",
    "SizeCreateResponse",
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
    "GenerateSettings",
    "GenerateResponse",
]
