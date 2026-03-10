"""
Router for climb generation.

Endpoints:
- POST /walls/{wall_id}/generate  — Generate climbs using the DDPM
"""
from fastapi import APIRouter, HTTPException, Query

from app.schemas.generate import GenerateRequest, GenerateResponse, GenerateSettings, GradeScale
from app.services import services

router = APIRouter()


@router.get(
    "",
    response_model=GenerateResponse,
    summary="Generate climbs",
    description="Generate climbs for a wall using the pre-trained DDPM model.",
)
def generate_climbs(
    layout_id: str,
    num_climbs: int = Query(..., ge=1, le=10),
    grade: str = Query("V4"),
    grade_scale: GradeScale = Query(GradeScale.V_GRADE),
    angle: int | None = Query(None, ge=0, le=90),
    timesteps: int = Query(100, ge=1, le=100),
    t_start_projection: float = Query(1.0, ge=0.0, le=1.0),
    x_offset: float | None = Query(None, ge=-1.5, le=1.5),
    deterministic: bool = Query(False),
    seed: int | None = Query(None),
):
    """
    Generate climbs for a given wall.

    The model is a single pre-trained DDPM capable of generating for any
    wall. Holds are loaded from the database based on `wall_id`.

    Parameters:
    - **num_climbs**: How many climbs to generate (1–50)
    - **grade**: Target difficulty (e.g. 'V4', '6b+')
    - **grade_scale**: Grading system ('v_grade' or 'font')
    - **angle**: Wall angle override (defaults to the wall's stored angle)
    - **deterministic**: Fixed noise for reproducible output
    """
    request = GenerateRequest(
        num_climbs=num_climbs,
        grade=grade,
        grade_scale=grade_scale,
        angle=angle,
    )
    settings = GenerateSettings(
        timesteps=timesteps,
        t_start_projection=t_start_projection,
        deterministic=deterministic,
        x_offset=x_offset,
        seed=seed,
    )

    try:
        generated = services.generate_climbs(layout_id, request, settings)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    return GenerateResponse(
        wall_id=layout_id,
        climbs=generated,
        num_generated=len(generated),
        parameters=request,
    )
