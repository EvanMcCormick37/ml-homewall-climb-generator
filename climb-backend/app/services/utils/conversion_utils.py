from datetime import datetime
import json
import uuid
from app.schemas import SizeMetadata, LayoutMetadata, HoldDetail
from app.database import get_db

def _row_to_size_metadata(row) -> SizeMetadata:
    return SizeMetadata(
        id=row["id"],
        layout_id=row["layout_id"],
        name=row["name"],
        edges=json.loads(row["edges"]),
        kickboard=row["kickboard"],
        created_at=datetime.fromisoformat(row["created_at"])
            if isinstance(row["created_at"], str) else row["created_at"],
        updated_at=datetime.fromisoformat(row["updated_at"])
            if isinstance(row["updated_at"], str) else row["updated_at"],
    )


def _parse_sizes(rows) -> list[SizeMetadata]:
    """Convert size DB rows to SizeMetadata objects."""
    sizes = []
    for row in rows:
        sizes.append(_row_to_size_metadata(row))
    return sizes

def _row_to_layout_metadata(row, sizes: list[SizeMetadata]) -> LayoutMetadata:
    raw_corners = row["homography_src_corners"] if "homography_src_corners" in row.keys() else None
    return LayoutMetadata(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        dimensions=json.loads(row["dimensions"]),
        image_edges=json.loads(row["image_edges"]),
        homography_src_corners=json.loads(raw_corners) if raw_corners else None,
        default_angle=row["default_angle"],
        sizes=sizes,
        owner_id=row["owner_id"],
        visibility=row["visibility"],
        share_token=row["share_token"],
        created_at=datetime.fromisoformat(row["created_at"])
            if isinstance(row["created_at"], str) else row["created_at"],
        updated_at=datetime.fromisoformat(row["updated_at"])
            if isinstance(row["updated_at"], str) else row["updated_at"],
    )


def _hold_detail_to_row(layout_id: str, hold: HoldDetail) -> tuple:
    hold_id = f"hold-{uuid.uuid4().hex[:15]}"
    return (
        hold_id,
        layout_id,
        hold.hold_index,
        hold.x,
        hold.y,
        hold.pull_x or 0.0,
        hold.pull_y or -1.0,
        hold.useability or 1.0,
        hold.is_foot,
        json.dumps(hold.tags or []),
    )


def _row_to_hold_detail(row) -> HoldDetail:
    """Convert a database row to a HoldDetail object."""
    return HoldDetail(
        hold_index=row["hold_index"],
        x=row["x"],
        y=row["y"],
        pull_x=row["pull_x"],
        pull_y=row["pull_y"],
        useability=row["useability"],
        is_foot=row["is_foot"],
        tags=json.loads(row["tags"]) if row["tags"] else []
    )

def _get_layout_angle(layout_id: str, default_default_angle: int = 40) -> int:
    """
    Look up the default angle for a layout.
    Checks the layouts table first, then falls back to the legacy layouts table.
    """
    with get_db() as conn:
        row = conn.execute(
            "SELECT default_angle FROM layouts WHERE id = ?", (layout_id,)
        ).fetchone()
    return row["default_angle"] if (row and row["default_angle"] is not None) else default_default_angle

