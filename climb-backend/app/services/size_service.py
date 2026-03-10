"""
Service for managing sizes.

A size is a physical variant of a layout — it has its own photo and dimensions
but shares the layout's holdset (filtered to holds within its edge bounds).
"""
import uuid
from datetime import datetime
from pathlib import Path
import json

from fastapi import UploadFile

from app.database import get_db
from app.schemas.sizes import SizeMetadata, SizeCreate
from app.config import settings
from app.services.utils import _row_to_size_metadata


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
) -> str:
    """
    Create a new size for a layout, optionally uploading a photo.
    Returns the new size ID.
    """
    size_id = f"size-{uuid.uuid4().hex[:12]}"
    now = datetime.now()

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO sizes (id, layout_id, name, edges,
                               kickboard, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                size_id,
                layout_id,
                size_data.name,
                json.dumps(size_data.edges),
                size_data.kickboard,
                now,
                now,
            ),
        )
    return size_id


def delete_size(size_id: str) -> bool:
    """Delete a size."""
    with get_db() as conn:
        conn.execute("DELETE FROM sizes WHERE id = ?", (size_id,))

    return True