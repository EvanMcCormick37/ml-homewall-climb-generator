"""
Service for managing layouts (and their holds).

A layout is a unique hold arrangement. Multiple sizes can share one layout.
This module replaces wall_service.py for all new code; wall_service.py is
kept for the legacy /walls API until Phase 6 cleanup.
"""
import uuid
import json
from datetime import datetime
from pathlib import Path

from fastapi import UploadFile

from app.database import get_db
from app.schemas import (
    LayoutCreate,
    LayoutDetail,
    LayoutMetadata,
    HoldDetail,
)
from app.schemas.sizes import SizeMetadata
from app.config import settings


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_sizes(rows) -> list[SizeMetadata]:
    """Convert size DB rows to SizeMetadata objects."""
    sizes = []
    for row in rows:
        photo_url = None
        if row["photo_path"]:
            # Resolve legacy path (WALLS_DIR/layout_id/photo) or new path
            photo_url = row["photo_path"]  # will be resolved to full URL by router
        sizes.append(SizeMetadata(
            id=row["id"],
            layout_id=row["layout_id"],
            name=row["name"],
            width_ft=row["width_ft"],
            height_ft=row["height_ft"],
            edge_left=row["edge_left"] or 0.0,
            edge_right=row["edge_right"],
            edge_bottom=row["edge_bottom"] or 0.0,
            edge_top=row["edge_top"],
            photo_url=photo_url,
            num_climbs=row["num_climbs"] or 0,
            created_at=datetime.fromisoformat(row["created_at"])
                if isinstance(row["created_at"], str) else row["created_at"],
            updated_at=datetime.fromisoformat(row["updated_at"])
                if isinstance(row["updated_at"], str) else row["updated_at"],
        ))
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


def _row_to_hold_detail(row) -> HoldDetail:
    return HoldDetail(
        hold_index=row["hold_index"],
        x=row["x"],
        y=row["y"],
        pull_x=row["pull_x"],
        pull_y=row["pull_y"],
        useability=row["useability"],
        is_foot=row["is_foot"],
        tags=json.loads(row["tags"]) if row["tags"] else [],
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
        hold.is_foot,
        json.dumps(hold.tags or []),
    )


# ---------------------------------------------------------------------------
# Layout queries
# ---------------------------------------------------------------------------

def layout_exists(layout_id: str) -> bool:
    with get_db() as conn:
        row = conn.execute(
            "SELECT 1 FROM layouts WHERE id = ?", (layout_id,)
        ).fetchone()
    return row is not None


def get_num_holds(layout_id: str) -> int | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT num_holds FROM layouts WHERE id = ?", (layout_id,)
        ).fetchone()
    return row["num_holds"] if row else None


def get_holds(layout_id: str, size_id: str | None = None) -> list[HoldDetail]:
    """
    Get holds for a layout. If size_id is given, filter to holds within that
    size's edge bounds.
    """
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT hold_index, x, y, pull_x, pull_y, useability, is_foot, tags
            FROM holds
            WHERE layout_id = ?
            ORDER BY hold_index ASC
            """,
            (layout_id,),
        ).fetchall()

        if size_id:
            size_row = conn.execute(
                "SELECT edge_left, edge_right, edge_bottom, edge_top FROM sizes WHERE id = ?",
                (size_id,),
            ).fetchone()
        else:
            size_row = None

    holds = [_row_to_hold_detail(row) for row in rows]

    if size_row:
        holds = _filter_holds_by_edges(holds, size_row)

    return holds


def _filter_holds_by_edges(holds: list[HoldDetail], size_row) -> list[HoldDetail]:
    """Return only holds within the size's edge bounds."""
    el = size_row["edge_left"]
    er = size_row["edge_right"]
    eb = size_row["edge_bottom"]
    et = size_row["edge_top"]
    return [
        h for h in holds
        if (el is None or h.x >= el)
        and (er is None or h.x <= er)
        and (eb is None or h.y >= eb)
        and (et is None or h.y <= et)
    ]


def get_all_layouts(owner_id: str | None = None) -> list[LayoutMetadata]:
    """Get all layouts with their sizes."""
    if owner_id is not None:
        where = "WHERE l.owner_id = ? OR l.visibility = 'public'"
        params = (owner_id,)
    else:
        where = "WHERE l.visibility = 'public'"
        params = ()

    with get_db() as conn:
        layout_rows = conn.execute(
            f"""
            SELECT l.id, l.name, l.description, l.num_holds,
                   l.owner_id, l.visibility, l.share_token,
                   l.created_at, l.updated_at
            FROM layouts l
            {where}
            ORDER BY l.created_at DESC
            """,
            params,
        ).fetchall()

        all_size_rows = conn.execute(
            "SELECT * FROM sizes ORDER BY created_at ASC"
        ).fetchall()

    # Group sizes by layout_id
    sizes_by_layout: dict[str, list] = {}
    for sr in all_size_rows:
        lid = sr["layout_id"]
        sizes_by_layout.setdefault(lid, []).append(sr)

    layouts = []
    for row in layout_rows:
        lid = row["id"]
        sizes = _parse_sizes(sizes_by_layout.get(lid, []))
        layouts.append(_row_to_layout_metadata(row, sizes))

    return layouts


def get_layout(layout_id: str, size_id: str | None = None) -> LayoutDetail | None:
    """Get full layout details including holds (optionally size-filtered)."""
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT id, name, description, num_holds, owner_id,
                   visibility, share_token, created_at, updated_at
            FROM layouts WHERE id = ?
            """,
            (layout_id,),
        ).fetchone()
        if not row:
            return None

        size_rows = conn.execute(
            "SELECT * FROM sizes WHERE layout_id = ? ORDER BY created_at ASC",
            (layout_id,),
        ).fetchall()

    sizes = _parse_sizes(size_rows)
    metadata = _row_to_layout_metadata(row, sizes)
    holds = get_holds(layout_id, size_id)
    return LayoutDetail(metadata=metadata, holds=holds)


def get_layout_visibility(layout_id: str) -> dict | None:
    """Fetch ownership/visibility fields for access checks."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, owner_id, visibility, share_token FROM layouts WHERE id = ?",
            (layout_id,),
        ).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Layout mutations
# ---------------------------------------------------------------------------

def create_layout(
    layout_data: LayoutCreate,
    owner_id: str,
) -> str:
    """Create a new layout (no photo/size — those come via create_size)."""
    layout_id = f"layout-{uuid.uuid4().hex[:12]}"
    share_token = uuid.uuid4().hex
    now = datetime.now()

    # Create layout directory
    layout_dir = settings.LAYOUTS_DIR / layout_id
    layout_dir.mkdir(parents=True, exist_ok=True)

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO layouts (id, name, description, num_holds, owner_id,
                                 visibility, share_token, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                layout_id,
                layout_data.name,
                layout_data.description,
                0,
                owner_id,
                layout_data.visibility,
                share_token,
                now,
                now,
            ),
        )
    return layout_id


def delete_layout(layout_id: str) -> bool:
    """Delete a layout and all associated sizes, holds, and climbs."""
    if not layout_exists(layout_id):
        return False

    with get_db() as conn:
        conn.execute("DELETE FROM holds WHERE layout_id = ?", (layout_id,))
        conn.execute("DELETE FROM layouts WHERE id = ?", (layout_id,))

    # Remove layout directory if it exists
    import shutil
    layout_dir = settings.LAYOUTS_DIR / layout_id
    if layout_dir.exists():
        shutil.rmtree(layout_dir)
    # Also clean legacy WALLS_DIR path (for migrated layouts)
    legacy_dir = settings.WALLS_DIR / layout_id
    if legacy_dir.exists():
        shutil.rmtree(legacy_dir)

    return True


def set_holds(layout_id: str, holds: list[HoldDetail]) -> bool:
    """Set or replace the full hold set for a layout."""
    if not layout_exists(layout_id):
        return False

    with get_db() as conn:
        conn.execute("DELETE FROM holds WHERE layout_id = ?", (layout_id,))
        conn.executemany(
            """
            INSERT INTO holds (id, layout_id, wall_id, hold_index, x, y,
                               pull_x, pull_y, useability, is_foot, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [_hold_detail_to_row(layout_id, hold) for hold in holds],
        )
        conn.execute(
            "UPDATE layouts SET num_holds = ?, updated_at = ? WHERE id = ?",
            (len(holds), datetime.now(), layout_id),
        )
    return True


# ---------------------------------------------------------------------------
# Photo helpers (legacy path support)
# ---------------------------------------------------------------------------

def get_photo_path(layout_id: str, size_id: str | None = None) -> Path | None:
    """
    Get the photo path for a layout/size combo.

    - If size_id is provided: look up that size's photo_path.
    - Otherwise: use the first size's photo_path.
    - Falls back to legacy WALLS_DIR/layout_id/photo.jpg path.
    """
    with get_db() as conn:
        if size_id:
            row = conn.execute(
                "SELECT photo_path FROM sizes WHERE id = ? AND layout_id = ?",
                (size_id, layout_id),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT photo_path FROM sizes WHERE layout_id = ? ORDER BY created_at ASC LIMIT 1",
                (layout_id,),
            ).fetchone()

    if not row or not row["photo_path"]:
        return None

    filename = row["photo_path"]

    # New-style path: LAYOUTS_DIR/layout_id/sizes/size_id/photo.ext
    if size_id:
        new_path = settings.LAYOUTS_DIR / layout_id / "sizes" / size_id / filename
        if new_path.exists():
            return new_path

    # Legacy path: WALLS_DIR/layout_id/photo.ext  (migrated walls)
    legacy_path = settings.WALLS_DIR / layout_id / filename
    if legacy_path.exists():
        return legacy_path

    return None
