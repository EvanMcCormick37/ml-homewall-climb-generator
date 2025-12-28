"""Basic pydantic schemas used for building larger input schemas"""
from pydantic import BaseModel, Field
from typing import Annotated

# Climbing Position Annotations
PositiveInt = Annotated[int,Field(ge=0)]
HoldPosition = Annotated[list[PositiveInt],Field(
        ...,
        min_length=2, 
        max_length=2,
        description="[left_hand_hold_id, right_hand_hold_id]"
    )]

# --- Hold Base ---

class HoldDetail(BaseModel):
    """Base hold schema."""
    hold_id: PositiveInt
    norm_x: float = Field(..., ge=0, le=1)
    norm_y: float = Field(..., ge=0, le=1)
    pull_x: float = Field(..., ge=-1, le=1)
    pull_y: float = Field(..., ge=-1, le=1)
    useability: float | None = Field(None, ge=0, le=10)

class Holdset(BaseModel):
    """Schema for hold sets for a climb"""
    start: HoldPosition
    finish: HoldPosition
    hand: list[PositiveInt]
    foot: list[PositiveInt]