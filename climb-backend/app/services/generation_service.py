"""
Service for generating climbs using the pre-trained DDPM.

Manages:
- Lazy initialization of the DDPM generator singleton
- Wall angle lookup for default angles
- Dispatching generation requests
"""
import logging

from app.config import settings
from app.schemas import Holdset, GenerateRequest
from app.services.utils import (
    ClimbDDPM,
    ClimbDDPMGenerator,
    ClimbsFeatureScaler,
    Noiser,
    UNetHoldClassifierLogits
)
from app.services.climb_service import _holds_to_holdset

logger = logging.getLogger(__name__)

def generate_climbs(
    wall_id: str,
    request: GenerateRequest,
) -> list[Holdset]:
    """
    Generate climbs for a wall using the DDPM.

    Args:
        wall_id: Target wall ID (holds loaded from DB)
        request: Generation parameters

    Returns:
        List of GeneratedClimb results
    """
    ddpm = ClimbDDPM(
        model=Noiser(),
        weights_path=settings.DDPM_WEIGHTS_PATH,
        timesteps=100,
    )
    scaler = ClimbsFeatureScaler(
        weights_path=settings.SCALER_WEIGHTS_PATH
    )
    hold_classifier = UNetHoldClassifierLogits(
        weights_path=settings.HC_WEIGHTS_PATH
    )
    generator = ClimbDDPMGenerator(
        wall_id=wall_id,
        scaler=scaler,
        ddpm=ddpm,
        hold_classifier=hold_classifier
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

    return [_holds_to_holdset(c) for c in raw_climbs]
