"""
Service for managing walls.

Handles:
- Wall CRUD operations
- Hold data storage (JSON files)
- Photo storage
"""
import json
import uuid
import shutil
from pathlib import Path
from datetime import datetime

from fastapi import UploadFile

from app.database import get_db, WALLS_DIR
from app.schemas import WallCreate, WallDetail, WallMetadata, Hold


def _parse_dimensions(dim_str: str | None) -> tuple[int, int] | None:
    """Parse dimensions string 'width, height' into tuple."""
    if not dim_str:
        return None
    try:
        parts = dim_str.split(",")
        return (int(parts[0].strip()), int(parts[1].strip()))
    except (ValueError, IndexError):
        return None


class WallService:
    """Service for wall operations."""
    
    def _get_wall_dir(self, wall_id: str) -> Path:
        """Get the directory for a wall's files."""
        return WALLS_DIR / wall_id
    
    def _get_wall_json_path(self, wall_id: str) -> Path:
        """Get path to wall's JSON data file."""
        return self._get_wall_dir(wall_id) / "wall.json"
    
    def _get_photo_path(self, wall_id: str) -> Path:
        """Get path to wall's photo."""
        return self._get_wall_dir(wall_id) / "photo.jpg"
    
    def wall_exists(self, wall_id: str) -> bool:
        """Check if a wall exists."""
        with get_db() as conn:
            row = conn.execute(
                "SELECT 1 FROM walls WHERE id = ?", (wall_id,)
            ).fetchone()
        return row is not None
    
    def get_num_holds(self, wall_id: str) -> int | None:
        """Get the number of holds of a wall, if it exists"""
        with get_db() as conn:
            row = conn.execute(
                "SELECT num_holds FROM walls WHERE id = ?", (wall_id,)
            ).fetchone()
        return row["num_holds"] if row else None
    
    def get_all_walls(self) -> list[WallMetadata]:
        """Get all walls with basic metadata."""
        with get_db() as conn:
            # Get all walls with climb and model counts
            rows = conn.execute("""
                SELECT 
                    w.id,
                    w.name,
                    w.num_holds,
                    w.dimensions,
                    w.angle,
                    w.photo,
                    w.created_at,
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
                num_holds=row["num_holds"] or 0,
                num_climbs=row["num_climbs"],
                num_models=row["num_models"],
                dimensions=_parse_dimensions(row["dimensions"]),
                angle=row["angle"],
                photo_url=row["photo"],
                created_at=datetime.fromisoformat(row["created_at"]) 
                    if isinstance(row["created_at"], str) else row["created_at"],
            ))
        
        return walls
    
    def get_wall(self, wall_id: str) -> WallDetail | None:
        """Get full wall details including holds."""
        with get_db() as conn:
            # Get wall metadata with counts
            row = conn.execute("""
                SELECT 
                    w.id,
                    w.name,
                    w.num_holds,
                    w.dimensions,
                    w.angle,
                    w.photo,
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
        json_path = self._get_wall_json_path(wall_id)
        if json_path.exists():
            with open(json_path, "r") as f:
                wall_json = json.load(f)
                holds = [Hold(**hold_data) for hold_data in wall_json.get("holds", [])]
        
        return WallDetail(
            id=row["id"],
            name=row["name"],
            num_holds=row["num_holds"] or 0,
            num_climbs=row["num_climbs"],
            num_models=row["num_models"],
            dimensions=_parse_dimensions(row["dimensions"]),
            angle=row["angle"],
            holds=holds,
            photo_url=row["photo"],
            created_at=datetime.fromisoformat(row["created_at"]) 
                if isinstance(row["created_at"], str) else row["created_at"],
            updated_at=datetime.fromisoformat(row["updated_at"]) 
                if isinstance(row["updated_at"], str) else row["updated_at"],
        )
    
    def create_wall(
        self, 
        wall_data: WallCreate,
    ) -> str:
        """
        Create a new wall.
        
        Args:
            wall_data: Wall creation data with holds
            photo_url: URL/path to the wall photo
            dimensions: Optional wall dimensions (width, height) in cm
            angle: Optional wall angle in degrees from vertical
            
        Returns:
            The new wall ID
        """
        wall_id = f"wall-{uuid.uuid4().hex[:12]}"
        wall_dir = self._get_wall_dir(wall_id)
        wall_dir.mkdir(parents=True, exist_ok=True)
        
        # Save holds to JSON
        holds_data = [hold.model_dump() for hold in wall_data.holds]
        created_at = datetime.now()
        wall_json = {
            "id": wall_id,
            "name": wall_data.name,
            "holds": holds_data,
        }
        
        with open(self._get_wall_json_path(wall_id), "w") as f:
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
                INSERT INTO walls (id, name, dimensions, angle, num_holds, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (wall_id, wall_data.name, dim_str, angle, len(wall_data.holds), created_at, created_at),
            )
        
        return wall_id
    
    def delete_wall(self, wall_id: str) -> bool:
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
        wall_dir = self._get_wall_dir(wall_id)
        if wall_dir.exists():
            shutil.rmtree(wall_dir)
        
        return True
    
    def get_photo_path(self, wall_id: str) -> Path | None:
        """Get path to wall photo if it exists."""
        path = self._get_photo_path(wall_id)
        return path if path.exists() else None
    
    async def save_photo(self, wall_id: str, photo: UploadFile) -> bool:
        """
        Save or replace wall photo.
        
        Args:
            wall_id: The wall ID
            photo: Uploaded photo file
            
        Returns:
            True if saved, False if wall not found
        """
        if not self.wall_exists(wall_id):
            return False
        
        photo_path = self._get_photo_path(wall_id)
        
        # Save file
        contents = await photo.read()
        with open(photo_path, "wb") as f:
            f.write(contents)
        
        return True