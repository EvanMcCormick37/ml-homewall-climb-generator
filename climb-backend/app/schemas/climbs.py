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
    holds: Holdset
    grade: int | None = Field(None, ge=0, le=180)
    setter: str | None = None
    tags: list[str] | None = None


class Climb(BaseModel):
    """Schema for climb response."""
    id: str
    wall_id: str
    name: str | None = None
    grade: int | None = None
    setter: str | None = None
    sequence: list[list[int]]
    tags: list[str] | None = None
    num_moves: int
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
