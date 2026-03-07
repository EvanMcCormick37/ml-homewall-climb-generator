"""
Service for managing sizes.

A size is a physical variant of a layout — it has its own photo and dimensions
but shares the layout's holdset (filtered to holds within its edge bounds).
"""
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import UploadFile

from app.database import get_db
from app.schemas.sizes import SizeMetadata, SizeCreate
from app.config import settings


def _row_to_size_metadata(row) -> SizeMetadata:
    return SizeMetadata(
        id=row["id"],
        layout_id=row["layout_id"],
        name=row["name"],
        width_ft=row["width_ft"],
        height_ft=row["height_ft"],
        edge_left=row["edge_left"] or 0.0,
        edge_right=row["edge_right"],
        edge_bottom=row["edge_bottom"] or 0.0,
        edge_top=row["edge_top"],
        photo_url=row["photo_path"],  # resolved to full URL by router
        num_climbs=row["num_climbs"] or 0,
        created_at=datetime.fromisoformat(row["created_at"])
            if isinstance(row["created_at"], str) else row["created_at"],
        updated_at=datetime.fromisoformat(row["updated_at"])
            if isinstance(row["updated_at"], str) else row["updated_at"],
    )


def size_exists(size_id: str) -> bool:
    with get_db() as conn:
        row = conn.execute(
            "SELECT 1 FROM sizes WHERE id = ?", (size_id,)
        ).fetchone()
    return row is not None


def get_sizes(layout_id: str) -> list[SizeMetadata]:
    """Get all sizes for a layout."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM sizes WHERE layout_id = ? ORDER BY created_at ASC",
            (layout_id,),
        ).fetchall()
    return [_row_to_size_metadata(row) for row in rows]


def get_size(size_id: str) -> SizeMetadata | None:
    """Get a single size by ID."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM sizes WHERE id = ?", (size_id,)
        ).fetchone()
    return _row_to_size_metadata(row) if row else None


def create_size(
    layout_id: str,
    size_data: SizeCreate,
    photo: UploadFile | None = None,
) -> str:
    """
    Create a new size for a layout, optionally uploading a photo.
    Returns the new size ID.
    """
    size_id = f"size-{uuid.uuid4().hex[:12]}"
    now = datetime.now()

    photo_path_name = None
    if photo and photo.filename:
        photo_path_name = _save_size_photo(layout_id, size_id, photo)

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO sizes (id, layout_id, name, width_ft, height_ft,
                               edge_left, edge_right, edge_bottom, edge_top,
                               photo_path, num_climbs, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                size_id,
                layout_id,
                size_data.name,
                size_data.width_ft,
                size_data.height_ft,
                size_data.edge_left,
                size_data.edge_right,
                size_data.edge_bottom,
                size_data.edge_top,
                photo_path_name,
                0,
                now,
                now,
            ),
        )
    return size_id


def delete_size(size_id: str) -> bool:
    """Delete a size and its photo."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT layout_id, photo_path FROM sizes WHERE id = ?", (size_id,)
        ).fetchone()
        if not row:
            return False
        layout_id = row["layout_id"]
        conn.execute("DELETE FROM sizes WHERE id = ?", (size_id,))

    # Remove photo directory
    size_dir = settings.LAYOUTS_DIR / layout_id / "sizes" / size_id
    if size_dir.exists():
        import shutil
        shutil.rmtree(size_dir)

    return True


def upload_size_photo(layout_id: str, size_id: str, photo: UploadFile) -> bool:
    """Upload or replace the photo for a size."""
    if not size_exists(size_id):
        return False

    # Delete existing photo if any
    with get_db() as conn:
        row = conn.execute(
            "SELECT photo_path FROM sizes WHERE id = ?", (size_id,)
        ).fetchone()
        if row and row["photo_path"]:
            old_path = settings.LAYOUTS_DIR / layout_id / "sizes" / size_id / row["photo_path"]
            if old_path.exists():
                old_path.unlink()

    photo_path_name = _save_size_photo(layout_id, size_id, photo)

    with get_db() as conn:
        conn.execute(
            "UPDATE sizes SET photo_path = ?, updated_at = ? WHERE id = ?",
            (photo_path_name, datetime.now(), size_id),
        )
    return True


def get_photo_path(layout_id: str, size_id: str) -> Path | None:
    """Resolve the full filesystem path for a size's photo."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT photo_path FROM sizes WHERE id = ? AND layout_id = ?",
            (size_id, layout_id),
        ).fetchone()

    if not row or not row["photo_path"]:
        return None

    filename = row["photo_path"]

    # New-style path
    new_path = settings.LAYOUTS_DIR / layout_id / "sizes" / size_id / filename
    if new_path.exists():
        return new_path

    # Legacy path (migrated walls: photo lives at WALLS_DIR/layout_id/photo.ext)
    legacy_path = settings.WALLS_DIR / layout_id / filename
    if legacy_path.exists():
        return legacy_path

    return None


def _save_size_photo(layout_id: str, size_id: str, photo: UploadFile) -> str:
    """Save photo to LAYOUTS_DIR/layout_id/sizes/size_id/ and return filename."""
    assert photo.filename
    ext = Path(photo.filename).suffix
    size_dir = settings.LAYOUTS_DIR / layout_id / "sizes" / size_id
    size_dir.mkdir(parents=True, exist_ok=True)
    filename = f"photo{ext}"
    contents = photo.file.read()
    with open(size_dir / filename, "wb") as f:
        f.write(contents)
    return filename
