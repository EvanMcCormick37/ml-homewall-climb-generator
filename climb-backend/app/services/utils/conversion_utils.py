from datetime import datetime
import json
import uuid
from app.schemas import SizeMetadata, LayoutMetadata, HoldDetail,

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
    return LayoutMetadata(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        num_holds=row["num_holds"] or 0,
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
        layout_id,   # also set wall_id for backward compat
        hold.hold_index,
        hold.x,
        hold.y,
        hold.pull_x,
        hold.pull_y,
        hold.useability,
        json.dumps(hold.tags or []),
    )

def _parse_dimensions(dim_str: str | None = None) -> tuple[int, int] | None:
    """Parse dimensions string 'width, height' into tuple."""
    if not dim_str:
        return None
    try:
        parts = dim_str.split(",")
        return (int(parts[0].strip()), int(parts[1].strip()))
    except (ValueError, IndexError):
        return None


def _row_to_hold_detail(row) -> HoldDetail:
    """Convert a database row to a HoldDetail object."""
    return HoldDetail(
        hold_index=row["hold_index"],
        x=row["x"],
        y=row["y"],
        pull_x=row["pull_x"],
        pull_y=row["pull_y"],
        useability=row["useability"],
        tags=json.loads(row["tags"]) if row["tags"] else []
    )


def _hold_detail_to_row(wall_id: str, hold: HoldDetail) -> tuple:
    """Convert a HoldDetail object to database row values."""
    hold_id = f"hold-{uuid.uuid4().hex[:15]}"
    return (
        hold_id,
        wall_id,
        hold.hold_index,
        hold.x,
        hold.y,
        hold.pull_x,
        hold.pull_y,
        hold.useability,
        hold.is_foot,
        json.dumps(hold.tags or []),
    )

