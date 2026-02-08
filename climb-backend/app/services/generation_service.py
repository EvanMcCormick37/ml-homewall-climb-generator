"""
Service for generating climbs using the pre-trained DDPM.

Manages:
- Lazy initialization of the DDPM generator singleton
- Wall angle lookup for default angles
- Dispatching generation requests
"""
import logging

from app.database import get_db
from app.config import settings
from app.schemas.generate import GenerateRequest, GeneratedClimb
from app.services.utils.ddpm import (
    ClimbDDPM,
    ClimbDDPMGenerator,
    ClimbsFeatureScaler,
    Noiser,
)

logger = logging.getLogger(__name__)

# Module-level singleton — created on first use
_generator: ClimbDDPMGenerator | None = None


def _get_generator() -> ClimbDDPMGenerator:
    """Lazy-init the DDPM generator (loads weights once)."""
    global _generator
    if _generator is None:
        logger.info("Initializing DDPM generator (first request)...")
        model = ClimbDDPM(
            model=Noiser(
                hidden_dim=settings.DDPM_HIDDEN_DIM,
                layers=settings.DDPM_LAYERS,
                sinusoidal=settings.DDPM_SINUSOIDAL,
            ),
            timesteps=settings.DDPM_TIMESTEPS,
        )
        scaler = ClimbsFeatureScaler()
        _generator = ClimbDDPMGenerator(
            db_path=str(settings.DB_PATH),
            scaler=scaler,
            model=model,
            model_weights_path=str(settings.DDPM_WEIGHTS_PATH),
            scaler_weights_path=str(settings.DDPM_SCALER_PATH),
        )
        logger.info("DDPM generator ready.")
    return _generator


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
    generator = _get_generator()

    # Resolve angle: use request override, else wall's stored angle
    angle = request.angle
    if angle is None:
        angle = _get_wall_angle(wall_id)

    raw_climbs = generator.generate(
        wall_id=wall_id,
        n=request.num_climbs,
        angle=angle,
        grade=request.grade,
        diff_scale=request.grade_scale.value,
        deterministic=request.deterministic
    )

    return [
        GeneratedClimb(holds=holds, num_holds=len(holds))
        for holds in raw_climbs
    ]
