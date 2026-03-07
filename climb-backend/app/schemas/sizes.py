"""
Pydantic schemas for size-related requests and responses.

A "size" is a physical variant of a layout — different dimensions and photo,
but the same holdset (filtered by edge bounds).
"""
from pydantic import BaseModel, Field
from datetime import datetime


class SizeMetadata(BaseModel):
    """Size metadata."""
    id: str
    layout_id: str
    name: str
    width_ft: float | None = None
    height_ft: float | None = None
    edge_left: float = 0.0
    edge_right: float | None = None
    edge_bottom: float = 0.0
    edge_top: float | None = None
    photo_url: str | None = None
    num_climbs: int = 0
    created_at: datetime
    updated_at: datetime


class SizeCreate(BaseModel):
    """Schema for creating a new size."""
    name: str = Field(..., min_length=1, max_length=100)
    width_ft: float | None = None
    height_ft: float | None = None
    edge_left: float = Field(0.0, ge=0)
    edge_right: float | None = None
    edge_bottom: float = Field(0.0, ge=0)
    edge_top: float | None = None


class SizeCreateResponse(BaseModel):
    """Response after creating a size."""
    id: str
    layout_id: str
    name: str
