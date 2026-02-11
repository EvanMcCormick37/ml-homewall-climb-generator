"""Basic pydantic schemas used for building larger input schemas"""
from pydantic import BaseModel, Field
from typing import Annotated

# Climbing Position Annotations
PositiveInt = Annotated[int,Field(ge=0)]
HoldPosition = Annotated[list[PositiveInt],Field(
        ...,
        min_length=1, 
        max_length=2,
    )]

# --- Hold Base ---

class HoldDetail(BaseModel):
    """Base hold schema."""
    hold_index: int = Field(...,ge=0)
    x: float = Field(..., ge=0)
    y: float = Field(..., ge=0)
    pull_x: float | None = Field(None, ge=-1, le=1)
    pull_y: float | None = Field(None, ge=-1, le=1)
    useability: float | None = Field(None, ge=0, le=1)
    is_foot: int = Field(0, ge=0, le=1)

class Holdset(BaseModel):
    """Schema for hold sets for a climb"""
    start: HoldPosition
    finish: HoldPosition
    hand: list[PositiveInt]
    foot: list[PositiveInt]