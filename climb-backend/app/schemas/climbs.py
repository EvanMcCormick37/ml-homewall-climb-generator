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

class ClimbCreate(BaseModel):
    """Schema for creating a climb."""
    name: str
    holdset: Holdset
    angle: int
    grade: int | None = Field(None, ge=0, le=180)
    setter_name: str | None = None
    tags: list[str] | None = None


class Climb(BaseModel):
    """Schema for climb response."""
    id: str
    wall_id: str
    angle: int
    name: str
    grade: int | None = None
    setter_name: str | None = None
    holdset: Holdset
    tags: list[str] | None = None
    ascents: int
    created_at: datetime


class ClimbListResponse(BaseModel):
    """Response for listing climbs."""
    climbs: list[Climb]
    total: int
    limit: int
    offset: int


class ClimbCreateResponse(BaseModel):
    """Response after creating a climb."""
    id: str


class ClimbDeleteResponse(BaseModel):
    """Response after deleting a climb."""
    id: str
