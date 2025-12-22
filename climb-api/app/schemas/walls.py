"""
Pydantic schemas for wall-related requests and responses.
"""
from pydantic import BaseModel, Field
from fastapi import UploadFile, File
from datetime import datetime


# --- Hold Schemas ---

class HoldBase(BaseModel):
    """Base hold schema."""
    hold_id: int
    norm_x: float = Field(..., ge=0, le=1)
    norm_y: float = Field(..., ge=0, le=1)
    pull_x: float = Field(..., ge=-1, le=1)
    pull_y: float = Field(..., ge=-1, le=1)
    useability: float | None = Field(None, ge=0, le=10)


class HoldCreate(HoldBase):
    """Schema for creating a hold."""
    pass


class Hold(HoldBase):
    """Schema for hold response."""
    pass


# --- Wall Schemas ---

class WallMetadata(BaseModel):
    """Wall metadata without holds."""
    id: str
    name: str
    num_holds: int
    num_climbs: int = 0
    num_models: int = 0
    dimensions: tuple[int, int] | None = None
    angle: int | None = None
    photo_url: str
    created_at: datetime


class WallCreate(BaseModel):
    """Schema for creating a wall."""
    name: str = Field(..., min_length=1, max_length=100)
    holds: list[HoldCreate]
    dimensions: tuple[int, int] | None = None
    angle: int | None = None


class WallDetail(BaseModel):
    """Detailed wall info including holds."""
    id: str
    name: str
    num_holds: int
    num_climbs: int = 0
    num_models: int = 0
    dimensions: tuple[int, int] | None = None
    angle: int | None = None
    holds: list[Hold]
    photo_url: str
    created_at: datetime
    updated_at: datetime


class WallListResponse(BaseModel):
    """Response for listing walls."""
    walls: list[WallMetadata]
    total: int


class WallCreateResponse(BaseModel):
    """Response after creating a wall."""
    id: str
    name: str