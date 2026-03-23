"""
Service for generating climbs using the pre-trained DDPM.

Manages:
- Lazy initialization of the DDPM generator singleton
- Angle lookup for default angles
- Dispatching generation requests
"""
import logging

from app.schemas import Holdset, GenerateRequest, GenerateSettings
from app.database import get_db
from app.services.utils import generator
from app.services.climb_service import _holds_to_holdset, _get_layout_angle

logger = logging.getLogger(__name__)


def _get_layout_angle(layout_id: str, default_default_angle: int = 45) -> int:
    """
    Look up the default angle for a layout.
    Checks the layouts table first, then falls back to the legacy layouts table.
    """
    with get_db() as conn:
        row = conn.execute(
            "SELECT default_angle FROM layouts WHERE id = ?", (layout_id,)
        ).fetchone()
    return row["default_angle"] if (row and row["default_angle"] is not None) else default_default_angle


def generate_climbs(
    layout_id: str,
    request: GenerateRequest,
    gen_settings: GenerateSettings,
    size_id: str | None = None,
) -> list[Holdset]:
    """
    Generate climbs for a layout using the DDPM.

    Args:
        layout_id: Target layout ID (holds loaded from DB by the generator).
                   For migrated layouts this equals the old layout_id.
        request: Generation parameters (grade, angle, num_climbs, etc.)
        gen_settings: DDPM hyper-parameters.
        size_id: Optional size ID. Currently used only to log context;
                 full size-aware manifold filtering is a planned enhancement
                 (see TRICKY_DECISIONS.md).

    Returns:
        List of Holdset results.
    """
    try:
        # Resolve angle: use request override, else layout's stored angle
        angle = request.angle if request.angle else _get_layout_angle(layout_id)

        # The generator's internal lookup is keyed by layout_id.
        # For migrated data, layout_id == layout_id, so this still works.
        raw_climbs = generator.generate(
            layout_id=layout_id,
            n=request.num_climbs,
            angle=angle,
            grade=request.grade,
            diff_scale=request.grade_scale.value,
            timesteps=gen_settings.timesteps,
            guidance_value=gen_settings.guidance_value,
            deterministic=gen_settings.deterministic,
        )
    except Exception as e:
        print(f"Exception: {e}")
        raise e

    return [_holds_to_holdset(c) for c in raw_climbs]
