"""
Service for managing walls.

Handles:
- Wall CRUD operations
- Hold data storage (SQLite holds table)
- Photo storage
"""
import uuid
import shutil
from pathlib import Path
from datetime import datetime

from fastapi import UploadFile

from app.database import get_db
from app.schemas import WallCreate, WallDetail, WallMetadata, HoldDetail
from app.config import settings


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
        is_foot=row["is_foot"]
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
        None,  # tags - not in HoldDetail schema yet
    )


def wall_exists(wall_id: str) -> bool:
    """Check if a wall exists."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT 1 FROM walls WHERE id = ?", (wall_id,)
        ).fetchone()
    return row is not None


def get_num_holds(wall_id: str) -> int | None:
    """Get the number of holds of a wall, if it exists"""
    with get_db() as conn:
        row = conn.execute(
            "SELECT num_holds FROM walls WHERE id = ?", (wall_id,)
        ).fetchone()
    return row["num_holds"] if row else None


def get_holds(wall_id: str) -> list[HoldDetail]:
    """Get all holds for a wall from the database."""
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT hold_index, x, y, pull_x, pull_y, useability, is_foot
            FROM holds
            WHERE wall_id = ?
            ORDER BY hold_index ASC
            """,
            (wall_id,)
        ).fetchall()
    return [_row_to_hold_detail(row) for row in rows]


def get_all_walls() -> list[WallMetadata]:
    """Get all walls with basic metadata."""
    with get_db() as conn:
        # Get all walls with climb and model counts
        rows = conn.execute("""
            SELECT 
                w.id,
                w.name,
                w.photo_path,
                w.dimensions,
                w.angle,
                w.num_holds,
                w.created_at,
                w.updated_at,
                COUNT(DISTINCT c.id) AS num_climbs,
                COUNT(DISTINCT m.id) AS num_models
            FROM walls w
            LEFT JOIN climbs c ON c.wall_id = w.id
            LEFT JOIN models m ON m.wall_id = w.id
            GROUP BY w.id
            ORDER BY w.created_at DESC
        """).fetchall()
    
    walls = []
    for row in rows:
        walls.append(WallMetadata(
            id=row["id"],
            name=row["name"],
            photo_url=row["photo_path"],
            num_holds=row["num_holds"] or 0,
            num_climbs=row["num_climbs"],
            num_models=row["num_models"],
            dimensions=_parse_dimensions(row["dimensions"]),
            angle=row["angle"],
            created_at=datetime.fromisoformat(row["created_at"]) 
                if isinstance(row["created_at"], str) else row["created_at"],
            updated_at=datetime.fromisoformat(row["updated_at"]) 
                if isinstance(row["updated_at"], str) else row["updated_at"],
        ))
    
    return walls


def get_wall(wall_id: str) -> WallDetail | None:
    """Get full wall details including holds."""
    with get_db() as conn:
        # Get wall metadata with counts
        row = conn.execute("""
            SELECT 
                w.id,
                w.name,
                w.photo_path,
                w.num_holds,
                w.dimensions,
                w.angle,
                w.created_at,
                w.updated_at,
                COUNT(DISTINCT c.id) AS num_climbs,
                COUNT(DISTINCT m.id) AS num_models
            FROM walls w
            LEFT JOIN climbs c ON c.wall_id = w.id
            LEFT JOIN models m ON m.wall_id = w.id
            WHERE w.id = ?
            GROUP BY w.id
        """, (wall_id,)).fetchone()
    
    if not row:
        return None
    
    # Load holds from database
    holds = get_holds(wall_id)
    
    metadata = WallMetadata(
        id=row["id"],
        name=row["name"],
        photo_url=row["photo_path"],
        num_holds=row["num_holds"] or 0,
        num_climbs=row["num_climbs"],
        num_models=row["num_models"],
        dimensions=_parse_dimensions(row["dimensions"]),
        angle=row["angle"],
        created_at=datetime.fromisoformat(row["created_at"]) 
            if isinstance(row["created_at"], str) else row["created_at"],
        updated_at=datetime.fromisoformat(row["updated_at"]) 
            if isinstance(row["updated_at"], str) else row["updated_at"],
    )
    return WallDetail(
        metadata=metadata,
        holds=holds
    )


def create_wall(
    wall_data: WallCreate,
    photo: UploadFile
) -> str:
    """
    Create a new wall.
    
    Args:
        wall_data: Wall creation data
        photo: UploadFile containing wall photo (JPEG or PNG)
        
    Returns:
        The new wall ID
    """
    # Create wall-id and wall-id subdirectory
    wall_id = f"wall-{uuid.uuid4().hex[:12]}"
    wall_dir = settings.WALLS_DIR / wall_id
    wall_dir.mkdir(parents=True, exist_ok=True)

    # Save photo in wall_dir and remember photo path for later
    ext = Path(photo.filename).suffix
    photo_path = wall_dir / f"photo{ext}"
    contents = photo.file.read()
    with open(photo_path, "wb") as f:
        f.write(contents)
    
    # Create variables for storage in DB
    dims = wall_data.dimensions
    dim_str = f"{dims[0]}, {dims[1]}" if dims else None

    angle = wall_data.angle if wall_data.angle else None
    created_at = datetime.now()
    
    # Insert into database
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO walls (id, name, photo_path, dimensions, angle, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (wall_id, wall_data.name, photo_path.name, dim_str, angle, created_at, created_at),
        )
    
    return wall_id


def delete_wall(wall_id: str) -> bool:
    """
    Delete a wall and all associated data.
    
    Args:
        wall_id: The wall ID
        
    Returns:
        True if deleted, False if not found
    """
    if not wall_exists(wall_id):
        return False
    
    # Delete from database (cascades to climbs/models due to FK)
    # Also delete holds explicitly since they may not have FK cascade
    with get_db() as conn:
        conn.execute("DELETE FROM holds WHERE wall_id = ?", (wall_id,))
        conn.execute("DELETE FROM walls WHERE id = ?", (wall_id,))
    
    # Delete wall directory (photo, model files)
    wall_dir = settings.WALLS_DIR / wall_id
    if wall_dir.exists():
        shutil.rmtree(wall_dir)
    
    return True


def set_holds(wall_id: str, holds: list[HoldDetail]) -> bool:
    """
    Set or replace holds for a wall.
    
    Args:
        wall_id: The wall ID
        holds: List of HoldDetail objects
        
    Returns:
        True if successful, False if wall not found
    """
    if not wall_exists(wall_id):
        return False
    
    with get_db() as conn:
        # Delete existing holds for this wall
        conn.execute("DELETE FROM holds WHERE wall_id = ?", (wall_id,))
        
        # Insert new holds
        conn.executemany(
            """
            INSERT INTO holds (id, wall_id, hold_index, x, y, pull_x, pull_y, useability, is_foot, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [_hold_detail_to_row(wall_id, hold) for hold in holds]
        )
        
        # Update the wall info in the database
        conn.execute(
            """
            UPDATE walls
            SET num_holds = ?, updated_at = ?
            WHERE id = ?
            """,
            (len(holds), datetime.now(), wall_id)
        )
    
    return True


def get_photo_path(wall_id: str) -> Path | None:
    """Get photo path for wall_id from the walls table"""
    if not wall_exists(wall_id):
        return None
    with get_db() as conn:
        row = conn.execute(
            "SELECT photo_path FROM walls WHERE id = ?", 
            (wall_id,)
        ).fetchone()
    return settings.WALLS_DIR / wall_id / row['photo_path']


def replace_photo(wall_id: str, photo: UploadFile) -> bool:
    """
    Replace wall photo by removing old versions and saving the new one.
    """
    if not wall_exists(wall_id):
        return False
    # Get the existing photo path and delete the current photo
    photo_path = get_photo_path(wall_id)
    if photo_path and photo_path.exists():
        photo_path.unlink()
    # Upload the image to the new photo path and save the new photo path to the database
    save_photo(wall_id, photo)
    return True


def save_photo(wall_id: str, photo: UploadFile):
    """Save a photo for a wall and update the database."""
    # Get the extension from the incoming file and create the new photo path
    ext = Path(photo.filename).suffix
    photo_path = settings.WALLS_DIR / wall_id / f"photo{ext}"
    # Save the new file
    contents = photo.file.read()
    with open(photo_path, "wb") as f:
        f.write(contents)
    
    # Add the new photo path to the database
    with get_db() as conn:
        conn.execute(
            "UPDATE walls SET photo_path = ?, updated_at = ? WHERE id = ?",
            (photo_path.name, datetime.now(), wall_id)
        )