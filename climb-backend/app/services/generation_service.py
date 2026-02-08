"""
Service for generating climbs using the pre-trained DDPM.

Manages:
- Lazy initialization of the DDPM generator singleton
- Wall angle lookup for default angles
- Dispatching generation requests
"""
import logging

from app.config import settings
from app.database import get_db
from app.schemas.generate import GenerateRequest, GeneratedClimb
from app.services.utils.ddpm import (
    ClimbDDPM,
    ClimbDDPMGenerator,
    ClimbsFeatureScaler,
    Noiser,
)

logger = logging.getLogger(__name__)

def _get_wall_angle(wall_id: str) -> int | None:
    """Look up the stored angle for a wall."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT angle FROM walls WHERE id = ?", (wall_id,)
        ).fetchone()
    return row["angle"] if row else None


def generate_climbs(
    wall_id: str,
    request: GenerateRequest,
) -> list[GeneratedClimb]:
    """
    Generate climbs for a wall using the DDPM.

    Args:
        wall_id: Target wall ID (holds loaded from DB)
        request: Generation parameters

    Returns:
        List of GeneratedClimb results
    """
    model = ClimbDDPM(
            model=Noiser(),
            weights_path=settings.DDPM_WEIGHTS_PATH,
            timesteps=100,
        )
    scaler = ClimbsFeatureScaler(
            weights_path=settings.SCALER_WEIGHTS_PATH
        )
    generator = ClimbDDPMGenerator(
        wall_id=wall_id,
        scaler=scaler,
        model=model
    )

    # Resolve angle: use request override, else wall's stored angle
    angle = request.angle

    raw_climbs = generator.generate(
        n=request.num_climbs,
        angle=angle,
        grade=request.grade,
        diff_scale=request.grade_scale.value,
        deterministic=request.deterministic
    )
    print(raw_climbs)

    return [
        GeneratedClimb(holds=holds, num_holds=len(holds))
        for holds in raw_climbs
    ]
