"""
Pydantic schemas for layout-related requests and responses.

A "layout" is a unique hold arrangement (what was previously called a "wall").
Multiple sizes can share the same layout/holdset.
"""
from pydantic import BaseModel, Field
from datetime import datetime
from app.schemas.base import HoldDetail
from app.schemas.sizes import SizeMetadata


class LayoutMetadata(BaseModel):
    """Layout metadata without holds."""
    id: str
    name: str
    description: str | None = None
    dimensions: list[int]
    image_edges: list[float]
    default_angle: int | None = None
    sizes: list[SizeMetadata] = []
    owner_id: str
    visibility: str
    share_token: str | None = None
    created_at: datetime
    updated_at: datetime


class LayoutCreate(BaseModel):
    """Schema for creating a layout (metadata only — photo comes via first size)."""
    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = None
    dimensions: list[int]
    default_angle: int | None = None
    image_edges: list[float]
    visibility: str = "public"

class LayoutEdit(BaseModel):
    name: str | None = None
    description: str | None = None
    dimensions: list[int] | None
    default_angle: int | None = None
    image_edges: list[float] | None = None
    visibility: str | None = None

class LayoutDetail(BaseModel):
    """Detailed layout info including holds and sizes."""
    metadata: LayoutMetadata
    holds: list[HoldDetail]


class LayoutListResponse(BaseModel):
    """Response for listing layouts."""
    layouts: list[LayoutMetadata]
    total: int


class LayoutCreateResponse(BaseModel):
    """Response after creating a layout."""
    id: str
    name: str


class SetHoldsResponse(BaseModel):
    """Response after setting holds on a layout."""
    id: str
