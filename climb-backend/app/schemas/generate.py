"""
Pydantic schemas for climb generation requests and responses.

Replaces the old model CRUD schemas — the backend now uses a single
pre-trained DDPM that generates for any layout.
"""
from pydantic import BaseModel, Field
from app.schemas.base import Holdset
from enum import Enum


class GradeScale(str, Enum):
    """Supported grading systems."""
    V_GRADE = "v_grade"
    FONT = "font"

class GenerateSettings(BaseModel):
    timesteps: int = Field(100, ge=1, le=200, description="Number of diffusion timesteps. Fewer = faster but lower quality.")
    guidance_value: float = Field(3.0, ge=1.0, le=10.0, description="CFG guidance scale. Higher = stronger grade/style conditioning.")
    t_start_projection: float = Field(0.8, ge=0.0, le=0.8, description = "The time at which to start the projection process.")
    deterministic: bool = Field(False, description="Reuse the initial noise vector at each step for reproducible output.")
    seed: int = Field(37, description="Seed value for deterministic generation.")

class GenerateRequest(BaseModel):
    """Request schema for generating climbs via the DDPM."""
    num_climbs: int = Field(..., ge=1, le=10, description="Number of climbs to generate")
    grade: str = Field("V4", description="Target difficulty grade (e.g. 'V4', '6b+')")
    grade_scale: GradeScale = Field(GradeScale.V_GRADE, description="Grading system to use")
    angle: int | None = Field(None, ge=0, le=90, description="Wall angle override (defaults to layout's stored angle)")
    x_offset: float | None = Field(None, ge=-1.5, le=1.5, description="X-offset to pin the climb to a specific horizontal position.")

class GenerateResponse(BaseModel):
    """Response containing generated climbs."""
    layout_id: str
    climbs: list[Holdset]
    num_generated: int
    parameters: GenerateRequest