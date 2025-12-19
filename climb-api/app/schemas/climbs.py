"""
Pydantic schemas for climb-related requests and responses.
"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


class ClimbSortBy(str, Enum):
    """Enum for climb sorting options."""
    DATE = "date"
    NAME = "name"
    GRADE = "grade"
    TICKS = "ticks"
    NUM_MOVES = "num_moves"

# --- Climb Schemas ---

class ClimbCreate(BaseModel):
    """Schema for creating a climb."""
    name: str | None = Field(None, max_length=100)
    grade: str | None = Field(None, max_length=10)
    setter: str | None = Field(None, max_length=50)
    sequence: list[list[int]] = Field(
        ..., 
        description="List of positions, each position is [lh_hold_id, rh_hold_id]"
    )
    tags: list[str] | None = None


class Climb(BaseModel):
    """Schema for climb response."""
    id: str
    wall_id: str
    name: str | None
    grade: str | None
    setter: str | None
    sequence: list[list[int]]
    tags: list[str] | None
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
    message: str = "Climb created successfully"


class ClimbDeleteResponse(BaseModel):
    """Response after deleting a climb."""
    id: str
    message: str = "Climb deleted successfully"
