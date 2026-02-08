"""
Pydantic schemas for climb generation requests and responses.

Replaces the old model CRUD schemas — the backend now uses a single
pre-trained DDPM that generates for any wall.
"""
from pydantic import BaseModel, Field
from enum import Enum


class GradeScale(str, Enum):
    """Supported grading systems."""
    V_GRADE = "v_grade"
    FONT = "font"


class GenerateRequest(BaseModel):
    """Request schema for generating climbs via the DDPM."""
    num_climbs: int = Field(5, ge=1, le=50, description="Number of climbs to generate")
    grade: str = Field("V4", description="Target difficulty grade (e.g. 'V4', '6b+')")
    grade_scale: GradeScale = Field(GradeScale.V_GRADE, description="Grading system to use")
    angle: int | None = Field(None, ge=0, le=90, description="Wall angle override (defaults to wall's stored angle)")
    deterministic: bool = Field(False, description="Use fixed noise for reproducible results")


class GeneratedClimb(BaseModel):
    """A single generated climb — an unordered set of hold indices."""
    holds: list[int]
    num_holds: int


class GenerateResponse(BaseModel):
    """Response containing generated climbs."""
    wall_id: str
    climbs: list[GeneratedClimb]
    num_generated: int
    parameters: GenerateRequest
