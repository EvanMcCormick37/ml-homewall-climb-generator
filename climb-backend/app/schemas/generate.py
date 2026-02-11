"""
Pydantic schemas for climb generation requests and responses.

Replaces the old model CRUD schemas — the backend now uses a single
pre-trained DDPM that generates for any wall.
"""
from pydantic import BaseModel, Field
from base import Holdset
from enum import Enum


class GradeScale(str, Enum):
    """Supported grading systems."""
    V_GRADE = "v_grade"
    FONT = "font"


class GenerateRequest(BaseModel):
    """Request schema for generating climbs via the DDPM."""
    num_climbs: int = Field(..., ge=1, le=50, description="Number of climbs to generate")
    grade: str = Field("V4", description="Target difficulty grade (e.g. 'V4', '6b+')")
    grade_scale: GradeScale = Field(GradeScale.V_GRADE, description="Grading system to use")
    angle: int = Field(..., ge=0, le=90, description="Wall angle override (defaults to wall's stored angle)")
    deterministic: bool = Field(False, description="Use fixed noise for reproducible results")

class GenerateResponse(BaseModel):
    """Response containing generated climbs."""
    wall_id: str
    climbs: list[Holdset]
    num_generated: int
    parameters: GenerateRequest
