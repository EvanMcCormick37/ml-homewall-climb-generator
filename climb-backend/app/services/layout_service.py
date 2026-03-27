"""
Service for managing layouts (and their holds).

A layout is a unique hold arrangement. Multiple sizes can share one layout.
"""
import json
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import UploadFile

from app.database import get_db
from app.schemas import (
    LayoutCreate,
    LayoutEdit,
    LayoutDetail,
    LayoutMetadata,
    HoldDetail,
)
from app.config import settings
from app.services.utils import _parse_sizes, _row_to_hold_detail, _hold_detail_to_row, _row_to_layout_metadata, generator_pool

# ---------------------------------------------------------------------------
# Layout queries
# ---------------------------------------------------------------------------

def layout_exists(layout_id: str) -> bool:
    with get_db() as conn:
        row = conn.execute(
            "SELECT 1 FROM layouts WHERE id = ?", (layout_id,)
        ).fetchone()
    return row is not None


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
                "SELECT edges FROM sizes WHERE id = ?",
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
    [l, r, b, t] = json.loads(size_row["edges"])
    return [
        h for h in holds
        if (l is None or h.x >= l)
        and (r is None or h.x <= r)
        and (b is None or h.y >= b)
        and (t is None or h.y <= t)
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
            SELECT l.id, l.name, l.description, l.dimensions, l.image_edges,
                   l.homography_src_corners, l.default_angle,
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

        climb_count_rows = conn.execute(
            "SELECT layout_id, COUNT(*) AS count FROM climbs GROUP BY layout_id"
        ).fetchall()

    # Group sizes by layout_id
    sizes_by_layout: dict[str, list] = {}
    for sr in all_size_rows:
        lid = sr["layout_id"]
        sizes_by_layout.setdefault(lid, []).append(sr)

    climb_counts: dict[str, int] = {r["layout_id"]: r["count"] for r in climb_count_rows}

    layouts = []
    for row in layout_rows:
        lid = row["id"]
        sizes = _parse_sizes(sizes_by_layout.get(lid, []))
        layouts.append(_row_to_layout_metadata(row, sizes, climb_counts.get(lid, 0)))

    return layouts


def get_layout(layout_id: str) -> LayoutDetail | None:
    """Get full layout details including holds (optionally size-filtered)."""
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT id, name, description, dimensions, image_edges, homography_src_corners,
                   default_angle, owner_id, visibility, share_token, created_at, updated_at
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
        climb_count_row = conn.execute(
            "SELECT COUNT(*) AS count FROM climbs WHERE layout_id = ?",(layout_id,)
        ).fetchone()

    sizes = _parse_sizes(size_rows)
    metadata = _row_to_layout_metadata(row, sizes, climb_count_row['count'])
    holds = get_holds(layout_id)
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
    """Create a new layout (no photo)."""
    layout_id = f"layout-{uuid.uuid4().hex[:12]}"
    share_token = uuid.uuid4().hex
    now = datetime.now()

    # Create layout directory
    layout_dir = settings.LAYOUTS_DIR / layout_id
    layout_dir.mkdir(parents=True, exist_ok=True)

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO layouts (id, name, description, dimensions, image_edges,
                                 homography_src_corners, default_angle, owner_id,
                                 visibility, share_token, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                layout_id,
                layout_data.name,
                layout_data.description,
                json.dumps(layout_data.dimensions),
                json.dumps(layout_data.image_edges),
                json.dumps(layout_data.homography_src_corners) if layout_data.homography_src_corners is not None else None,
                layout_data.default_angle,
                owner_id,
                layout_data.visibility,
                share_token,
                now,
                now,
            ),
        )

    generator_pool.update_all_hold_manifolds()
    
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
    
    generator_pool.update_all_hold_manifolds()

    return True


def put_layout(layout_id: str, layout_data: LayoutEdit) -> str:
    """Update an existing layout's metadata dynamically."""
    if not layout_exists(layout_id): #
        raise ValueError(f"Layout with id {layout_id} does not exist.")

    # 1. Initialize lists to hold our dynamic SQL pieces
    set_clauses = []
    params = []

    # 2. Check each field and add it if it's not None
    if layout_data.name is not None:
        set_clauses.append("name = ?")
        params.append(layout_data.name)
        
    if layout_data.description is not None:
        set_clauses.append("description = ?")
        params.append(layout_data.description)
        
    if layout_data.dimensions is not None:
        set_clauses.append("dimensions = ?")
        params.append(json.dumps(layout_data.dimensions)) #
        
    if layout_data.image_edges is not None:
        set_clauses.append("image_edges = ?")
        params.append(json.dumps(layout_data.image_edges))

    if layout_data.homography_src_corners is not None:
        set_clauses.append("homography_src_corners = ?")
        params.append(json.dumps(layout_data.homography_src_corners))

    if layout_data.default_angle is not None:
        set_clauses.append("default_angle = ?")
        params.append(layout_data.default_angle)
        
    if layout_data.visibility is not None:
        set_clauses.append("visibility = ?")
        params.append(layout_data.visibility)

    # 3. If everything was None, we don't need to hit the database at all
    if not set_clauses:
        return layout_id

    # 4. Always update the 'updated_at' timestamp if changes are being made
    set_clauses.append("updated_at = ?")
    params.append(datetime.now())

    # 5. Add the layout_id to the end of the parameters for the WHERE clause
    params.append(layout_id)

    # 6. Construct the final query string
    query = f"""
        UPDATE layouts 
        SET {", ".join(set_clauses)}
        WHERE id = ?
    """

    # 7. Execute the dynamic query
    with get_db() as conn: #
        conn.execute(query, tuple(params))
    
    generator_pool.update_all_hold_manifolds()
        
    return layout_id

def set_holds(layout_id: str, holds: list[HoldDetail]) -> bool:
    """Set or replace the full hold set for a layout."""
    if not layout_exists(layout_id):
        return False

    with get_db() as conn:
        conn.execute("DELETE FROM holds WHERE layout_id = ?", (layout_id,))
        conn.executemany(
            """
            INSERT INTO holds (id, layout_id, hold_index, x, y,
                               pull_x, pull_y, useability, is_foot, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [_hold_detail_to_row(layout_id, hold) for hold in holds],
        )
        conn.execute(
            "UPDATE layouts SET updated_at = ? WHERE id = ?",
            (datetime.now(), layout_id),
        )
    
    generator_pool.update_all_hold_manifolds()

    return True

def upload_layout_photo(layout_id: str, photo: UploadFile) -> bool:
    """Upload or replace the photo for a layout, and generate a 1/4 scale thumbnail."""
    from PIL import Image
    import io

    assert photo.filename

    ext = Path(photo.filename).suffix
    photo_dir = settings.LAYOUTS_DIR / layout_id
    photo_dir.mkdir(parents=True, exist_ok=True)

    contents = photo.file.read()

    # Save full-size photo
    with open(photo_dir / f"photo{ext}", "wb") as f:
        f.write(contents)

    # Generate and save 1/4 scale thumbnail
    img = Image.open(io.BytesIO(contents))
    small_size = (img.width // 4, img.height // 4)
    img_small = img.resize(small_size, Image.LANCZOS)
    img_small.save(photo_dir / f"photo-small{ext}")

    return True
