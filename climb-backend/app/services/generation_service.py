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
from app.database import get_db
from app.services.utils import (
    ClimbDDPM,
    ClimbDDPMGenerator,
    ClimbsFeatureScaler,
    Noiser,
    UNetHoldClassifierLogits
)
from app.services.climb_service import _holds_to_holdset

logger = logging.getLogger(__name__)

def _get_wall_angle(wall_id: str, default_angle: int = 45) -> int:
    """Get the default wall angle from the database. If there is no default wall angle, return 45."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT angle FROM walls WHERE id = ?", (wall_id,)
        ).fetchone()
    if row and row["angle"] is not None:
        return row["angle"]
    return default_angle

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
    print("Made it to Climb Generation!")

    ddpm = ClimbDDPM(
        model=Noiser(),
        weights_path=settings.DDPM_WEIGHTS_PATH,
        timesteps=100,
    )
    print("Instantiated ClimbDDPM!")
    scaler = ClimbsFeatureScaler(
        weights_path=settings.SCALER_WEIGHTS_PATH
    )
    print("Instantiated CFS!")
    hold_classifier = UNetHoldClassifierLogits(
        weights_path=settings.HC_WEIGHTS_PATH
    )
    print("Instantiated HoldClassifier!")
    generator = ClimbDDPMGenerator(
        wall_id=wall_id,
        scaler=scaler,
        ddpm=ddpm,
        hold_classifier=hold_classifier
    )

    print("Instantiated All Entities!")

    # Resolve angle: use request override, else wall's stored angle
    angle = request.angle if request.angle else _get_wall_angle(wall_id)

    print(f"Got Wall angle! {angle}. Running climb generation...")
    try:
        raw_climbs = generator.generate(
            n=request.num_climbs,
            angle=angle,
            grade=request.grade,
            diff_scale=request.grade_scale.value,
            deterministic=request.deterministic
        )
        print(raw_climbs)
    except Exception as e:
        print(f'Exception: {e}')
        raise e

    return [_holds_to_holdset(c) for c in raw_climbs]
