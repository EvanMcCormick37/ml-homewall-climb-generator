"""
Router for climb generation.
"""
from fastapi import APIRouter, HTTPException, Query

from app.schemas.generate import GenerateRequest, GenerateResponse, GenerateSettings
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
    difficulty: float = Query(..., description="Difficulty score (converted from grade on the client)"),
    angle: int | None = Query(None, ge=0, le=90),
    x_offset: float | None = Query(None),
    timesteps: int = Query(100, ge=1, le=200),
    t_start_projection: float = Query(0.8, le=0.8, ge=0.0),
    guidance_value: float = Query(3.0, ge=1.0, le=10.0),
    deterministic: bool = Query(False),
    seed: int = Query(37),
):
    request = GenerateRequest(
        num_climbs=num_climbs,
        difficulty=difficulty,
        angle=angle,
        x_offset=x_offset,
    )
    settings = GenerateSettings(
        timesteps=timesteps,
        guidance_value=guidance_value,
        deterministic=deterministic,
        t_start_projection=t_start_projection,
        seed=seed,
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


@router.get(
    "/evolutionary",
    response_model=GenerateResponse,
    summary="Generate climbs (evolutionary)",
    description=(
        "Generate climbs using an evolutionary diffusion strategy. "
        "`num_climbs` sets both the number of survivors kept per step (n_successors) "
        "and the number of climbs returned. `population` controls the total parallel "
        "candidates evaluated at each step."
    ),
)
def generate_climbs_evolutionary(
    layout_id: str,
    num_climbs: int = Query(..., ge=1, le=10),
    difficulty: float = Query(..., description="Difficulty score (converted from grade on the client)"),
    angle: int | None = Query(None, ge=0, le=90),
    x_offset: float | None = Query(None),
    population: int = Query(10, ge=2, le=10, description="Candidates per climb pool per diffusion step"),
    timesteps: int = Query(100, ge=1, le=200),
    guidance_value: float = Query(3.0, ge=1.0, le=10.0),
    deterministic: bool = Query(False),
    seed: int = Query(37),
):
    request = GenerateRequest(
        num_climbs=num_climbs,
        difficulty=difficulty,
        angle=angle,
        x_offset=x_offset,
    )
    settings = GenerateSettings(
        timesteps=timesteps,
        guidance_value=guidance_value,
        deterministic=deterministic,
        t_start_projection=0.5,  # unused in evolutionary mode
        seed=seed,
    )

    try:
        generated = services.generate_climbs_evolutionary(layout_id, request, settings, population)
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
