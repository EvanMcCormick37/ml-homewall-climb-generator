"""
Service for generating climbs using the pre-trained DDPM.

Manages:
- Lazy initialization of the DDPM generator singleton
- Wall angle lookup for default angles
- Dispatching generation requests
"""
import logging
from os import times

from app.schemas import Holdset, GenerateRequest, GenerateSettings
from app.database import get_db
from app.services.utils import generator, _get_wall_angle
from app.services.climb_service import _holds_to_holdset

logger = logging.getLogger(__name__)

def generate_climbs(
    wall_id: str,
    request: GenerateRequest,
    settings: GenerateSettings,
) -> list[Holdset]:
    """
    Generate climbs for a wall using the DDPM.

    Args:
        wall_id: Target wall ID (holds loaded from DB)
        request: Generation parameters

    Returns:
        List of GeneratedClimb results
    """

    # Resolve angle: use request override, else wall's stored angle
    angle = request.angle if request.angle else _get_wall_angle(wall_id)

    try:
        raw_climbs = generator.generate(
            wall_id=wall_id,
            n=request.num_climbs,
            angle=angle,
            grade=request.grade,
            diff_scale=request.grade_scale.value,
            timesteps=settings.timesteps,
            deterministic=settings.deterministic,
            t_start_projection=settings.t_start_projection,
            x_offset=settings.x_offset,
            seed=settings.seed if settings.seed else 37,
        )
    except Exception as e:
        print(f'Exception: {e}')
        raise e

    return [_holds_to_holdset(c) for c in raw_climbs]
