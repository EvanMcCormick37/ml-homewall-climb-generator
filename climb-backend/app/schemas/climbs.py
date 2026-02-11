"""
Pydantic schemas for climb-related requests and responses.
"""
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
from app.schemas.base import Holdset
class ClimbSortBy(str, Enum):
    """Enum for climb sorting options."""
    DATE = "date"
    NAME = "name"
    GRADE = "grade"
    ASCENTS = "ascents"

# --- Climb Schemas ---

class Climb(BaseModel):
    """Schema for climb response."""
    id: str
    wall_id: str
    angle: int
    name: str
    holdset: Holdset
    grade: float | None = None
    quality: float | None = None
    ascents: int
    setter_name: str | None = None
    tags: list[str] | None = None
    created_at: datetime

class ClimbCreate(BaseModel):
    """Schema for creating a climb."""
    name: str
    holdset: Holdset
    angle: int
    grade: float | None = Field(None, ge=0, le=39)
    quality: float | None = Field(2.5, ge=0, le=4)
    ascents: int | None = Field(0, ge=0)
    setter_name: str | None = None
    tags: list[str] | None = None

class ClimbBatchCreate(BaseModel):
    """Schema for batch creating climbs."""
    climbs: list[ClimbCreate]

class ClimbCreateResponse(BaseModel):
    """Response after creating a climb."""
    id: str

class ClimbBatchCreateResult(BaseModel):
    """Result for a single climb in batch creation."""
    index: int
    id: str | None = None
    status: str  # "success" or "error"
    error: str | None = None

class ClimbBatchCreateResponse(BaseModel):
    """Response after batch creating climbs."""
    total: int
    successful: int
    failed: int
    results: list[ClimbBatchCreateResult]

class ClimbDeleteResponse(BaseModel):
    """Response after deleting a climb."""
    id: str

class ClimbListResponse(BaseModel):
    """Response for listing climbs."""
    climbs: list[Climb]
    total: int
    limit: int
    offset: int