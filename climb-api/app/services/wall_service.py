"""
Service for managing walls.

Handles:
- Wall CRUD operations
- Hold data storage (JSON files)
- Photo storage
"""
import json
import uuid
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import UploadFile

from app.database import get_db, WALLS_DIR
from app.schemas import WallCreate, WallDetail, WallMetadata


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
    
    def get_all_walls(self) -> list[WallMetadata]:
        """Get all walls with basic metadata."""
        # TODO: Implement
        # Query walls table
        # For each wall, count climbs and models
        # Return list of WallMetadata
        raise NotImplementedError
    
    def get_wall(self, wall_id: str) -> Optional[WallDetail]:
        """Get full wall details including holds."""
        # TODO: Implement
        # 1. Query walls table for metadata
        # 2. Load holds from JSON file
        # 3. Count climbs and models
        # 4. Return WallDetail
        raise NotImplementedError
    
    def create_wall(self, wall_data: WallCreate) -> str:
        """
        Create a new wall.
        
        Args:
            wall_data: Wall creation data with holds
            
        Returns:
            The new wall ID
        """
        wall_id = f"wall-{uuid.uuid4().hex[:12]}"
        wall_dir = self._get_wall_dir(wall_id)
        wall_dir.mkdir(parents=True, exist_ok=True)
        
        # Save holds to JSON
        holds_data = [hold.model_dump() for hold in wall_data.holds]
        wall_json = {
            "id": wall_id,
            "name": wall_data.name,
            "holds": holds_data,
            "created_at": datetime.utcnow().isoformat(),
        }
        
        with open(self._get_wall_json_path(wall_id), "w") as f:
            json.dump(wall_json, f, indent=2)
        
        # Insert into database
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO walls (id, name, num_holds)
                VALUES (?, ?, ?)
                """,
                (wall_id, wall_data.name, len(wall_data.holds)),
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
        # TODO: Implement
        # 1. Check wall exists
        # 2. Delete from database (cascades to climbs/models)
        # 3. Delete wall directory (JSON, photo, model files)
        raise NotImplementedError
    
    def get_photo_path(self, wall_id: str) -> Optional[Path]:
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
