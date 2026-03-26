"""
Router for climb generation.

Endpoints:
- POST /layouts/{layout_id}/generate  — Generate climbs using the DDPM
"""
from fastapi import APIRouter, HTTPException, Query

from app.schemas.generate import GenerateRequest, GenerateResponse, GenerateSettings, GradeScale
from app.services import services

router = APIRouter()


@router.get(
    "",
    response_model=GenerateResponse,
    summary="Generate climbs",
    description="Generate climbs for a layout using the pre-trained DDPM model.",
)
def generate_climbs(
    layout_id: str,
    num_climbs: int = Query(..., ge=1, le=10),
    grade: str = Query("V4"),
    grade_scale: GradeScale = Query(GradeScale.V_GRADE),
    angle: int | None = Query(None, ge=0, le=90),
    timesteps: int = Query(100, ge=1, le=200),
    t_start_projection: float = Query(0.8, le=0.8, ge=0.0),
    x_offset: float | None = Query(None),
    guidance_value: float = Query(3.0, ge=1.0, le=10.0),
    deterministic: bool = Query(False),
    seed: int = Query(37)
):
    """
    Generate climbs for a given layout.
    """
    request = GenerateRequest(
        num_climbs=num_climbs,
        grade=grade,
        grade_scale=grade_scale,
        angle=angle,
    )
    settings = GenerateSettings(
        timesteps=timesteps,
        guidance_value=guidance_value,
        deterministic=deterministic,
        x_offset=x_offset,
        t_start_projection=t_start_projection,
        seed=seed
    )

    try:
        generated = services.generate_climbs(layout_id, request, settings)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    return GenerateResponse(
        layout_id=layout_id,
        climbs=generated,
        num_generated=len(generated),
        parameters=request,
    )
