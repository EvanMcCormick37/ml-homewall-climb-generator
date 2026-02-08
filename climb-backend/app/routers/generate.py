"""
Router for climb generation.

Endpoints:
- POST /walls/{wall_id}/generate  — Generate climbs using the DDPM
"""
from fastapi import APIRouter, HTTPException

from app.schemas.generate import GenerateRequest, GenerateResponse
from app.services import services

router = APIRouter()


@router.post(
    "",
    response_model=GenerateResponse,
    summary="Generate climbs",
    description="Generate climbs for a wall using the pre-trained DDPM model.",
)
def generate_climbs(wall_id: str, request: GenerateRequest):
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
    # Validate wall exists
    if not services.wall_exists(wall_id):
        raise HTTPException(status_code=404, detail="Wall not found")

    # Validate wall has holds
    num_holds = services.get_num_holds(wall_id)
    if not num_holds or num_holds == 0:
        raise HTTPException(
            status_code=400,
            detail="Wall has no holds. Upload holds before generating.",
        )

    try:
        generated = services.generate_climbs(wall_id, request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    return GenerateResponse(
        wall_id=wall_id,
        climbs=generated,
        num_generated=len(generated),
        parameters=request,
    )
