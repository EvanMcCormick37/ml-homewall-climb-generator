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
    edges: list[float]
    kickboard: bool
    created_at: datetime
    updated_at: datetime


class SizeCreate(BaseModel):
    """Schema for creating a new size."""
    name: str = Field(..., min_length=1, max_length=100)
    edges: list[float] = Field(...)
    kickboard: bool = Field(...)


class SizeCreateResponse(BaseModel):
    """Response after creating a size."""
    id: str
    layout_id: str
    name: str
