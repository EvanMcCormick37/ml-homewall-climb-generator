"""
Service for managing walls.

Handles:
- Wall CRUD operations
- HoldDetail data storage (JSON files)
- Photo storage
"""
import json
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

def wall_exists( wall_id: str) -> bool:
    """Check if a wall exists."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT 1 FROM walls WHERE id = ?", (wall_id,)
        ).fetchone()
    return row is not None

def get_num_holds( wall_id: str) -> int | None:
    """Get the number of holds of a wall, if it exists"""
    with get_db() as conn:
        row = conn.execute(
            "SELECT num_holds FROM walls WHERE id = ?", (wall_id,)
        ).fetchone()
    return row["num_holds"] if row else None

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

def get_wall( wall_id: str) -> WallDetail | None:
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
    
    # Load holds from JSON file
    holds = []
    json_path = settings.WALLS_DIR / wall_id / "wall.json"
    if json_path.exists():
        with open(json_path, "r") as f:
            wall_json = json.load(f)
            holds = [HoldDetail(**hold_data) for hold_data in wall_json.get("holds", [])]
    
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
        wall_data: Wall creation data with holds
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
    
    # Save holds to JSON in wall_dir
    holds_data = [hold.model_dump() for hold in wall_data.holds]
    created_at = datetime.now()
    wall_json = {
        "id": wall_id,
        "name": wall_data.name,
        "holds": holds_data,
    }
    
    with open(wall_dir / "wall.json", "w") as f:
        json.dump(wall_json, f, indent=2)
    
    # Serialize dimensions
    dims = wall_data.dimensions
    dim_str = f"{dims[0]}, {dims[1]}" if dims else None
    # Add wall angle if present
    angle = wall_data.angle if wall_data.angle else None
    
    # Insert into database
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO walls (id, name, photo_path, dimensions, angle, num_holds, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (wall_id, wall_data.name, photo_path.name, dim_str, angle, len(wall_data.holds), created_at, created_at),
        )
    
    return wall_id

def delete_wall( wall_id: str) -> bool:
    """
    Delete a wall and all associated data.
    
    Args:
        wall_id: The wall ID
        
    Returns:
        True if deleted, False if not found
    """
    
    # Delete from database (cascades to climbs/models due to FK)
    with get_db() as conn:
        conn.execute("DELETE FROM walls WHERE id = ?", (wall_id,))
    
    # Delete wall directory (JSON, photo, model files)
    wall_dir = settings.WALLS_DIR / wall_id
    if wall_dir.exists():
        shutil.rmtree(wall_dir)
        return True
    return False

def get_photo_path( wall_id: str) -> Path | None:
    """Get photo path for wall_id from the walls table"""
    if not wall_exists(wall_id):
        return None
    with get_db() as conn:
        row = conn.execute("SELECT photo_path FROM walls WHERE id = ?",(wall_id,)).fetchone()
    return settings.WALLS_DIR / wall_id / row['photo_path']
def replace_photo( wall_id: str, photo: UploadFile) -> bool:
    """
    Replace wall photo by removing old versions and saving the new one.
    """
    if not wall_exists(wall_id):
        return False
    # Get the existing photo path and delete the current photo
    photo_path = get_photo_path(wall_id)
    shutil.rmtree(photo_path)
    # Upload the image to the new photo path and save the new photo path to the database
    save_photo(wall_id, photo)
    return True

def save_photo( wall_id: str, photo: UploadFile):
    # Get the extension from the incoming file and create the new photo path
    ext = Path(photo.filename).suffix
    photo_path = settings.WALLS_DIR / wall_id / f"photo{ext}"
    # Save the new file
    contents = photo.file.read()
    with open(photo_path, "wb") as f:
        f.write(contents)
    
    # Add the new photo path to the database
    with get_db() as conn:
        conn.execute("UPDATE walls SET photo_path = ? WHERE id = ?",(photo_path,wall_id))