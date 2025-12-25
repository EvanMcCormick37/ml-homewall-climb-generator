"""
Pydantic schemas for wall-related requests and responses.
"""
from pydantic import BaseModel, Field
from datetime import datetime
from app.schemas.base import HoldDetail


# --- Wall Schemas ---

class WallMetadata(BaseModel):
    """Wall metadata without holds."""
    id: str
    name: str
    photo_url: str
    num_holds: int
    num_climbs: int = 0
    num_models: int = 0
    dimensions: tuple[int, int] | None = None
    angle: int | None = None
    created_at: datetime
    updated_at: datetime


class WallCreate(BaseModel):
    """Schema for creating a wall."""
    name: str = Field(..., min_length=1, max_length=100)
    dimensions: tuple[int, int] | None = None
    angle: int | None = None


class WallDetail(BaseModel):
    """Detailed wall info including holds."""
    metadata: WallMetadata
    holds: list[HoldDetail]


class WallListResponse(BaseModel):
    """Response for listing walls."""
    walls: list[WallMetadata]
    total: int


class WallCreateResponse(BaseModel):
    """Response after creating a wall."""
    id: str
    name: str

class SetHolds(BaseModel):
    """Schema for setting holds on an existing wall."""
    id: str
    holds: list[HoldDetail]

class SetHoldsResponse(BaseModel):
    """Response after setting holds on a wall."""
    id: str