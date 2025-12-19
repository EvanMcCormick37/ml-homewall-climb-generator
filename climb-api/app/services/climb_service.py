"""
Service for managing climbs.

Handles:
- Climb CRUD operations
- Filtering and sorting queries
"""
import json
import uuid
from datetime import datetime
from typing import Optional

from app.database import get_db
from app.schemas import Climb, ClimbCreate, ClimbSortBy


class ClimbService:
    """Service for climb operations."""
    
    def get_climbs(
        self,
        wall_id: str,
        name: Optional[str] = None,
        setter: Optional[str] = None,
        after: Optional[datetime] = None,
        sort_by: ClimbSortBy = ClimbSortBy.DATE,
        limit: int = 50,
        offset: int = 0,
        includes_holds: Optional[list[int]] = None,
    ) -> tuple[list[Climb], int]:
        """
        Get climbs for a wall with filtering.
        
        Args:
            wall_id: The wall ID
            name: Filter by name (partial match)
            setter: Filter by setter ID
            after: Filter climbs created after this date
            sort_by: Sort order
            limit: Maximum results
            offset: Pagination offset
            includes_holds: Hold IDs that must be in the climb
            
        Returns:
            Tuple of (list of climbs, total count before pagination)
        """
        # TODO: Implement
        # Build dynamic SQL query based on filters
        # Handle includes_holds by checking JSON sequence
        # Apply sorting and pagination
        raise NotImplementedError
    
    def get_climb(self, wall_id: str, climb_id: str) -> Optional[Climb]:
        """Get a single climb by ID."""
        with get_db() as conn:
            row = conn.execute(
                "SELECT * FROM climbs WHERE id = ? AND wall_id = ?",
                (climb_id, wall_id),
            ).fetchone()
        
        if not row:
            return None
        
        return self._row_to_climb(row)
    
    def create_climb(self, wall_id: str, climb_data: ClimbCreate) -> str:
        """
        Create a new climb.
        
        Args:
            wall_id: The wall ID
            climb_data: Climb data
            
        Returns:
            The new climb ID
        """
        climb_id = f"climb-{uuid.uuid4().hex[:12]}"
        
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO climbs (id, wall_id, name, grade, setter, sequence, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    climb_id,
                    wall_id,
                    climb_data.name,
                    climb_data.grade,
                    climb_data.setter,
                    json.dumps(climb_data.sequence),
                    json.dumps(climb_data.tags) if climb_data.tags else None,
                ),
            )
        
        return climb_id
    
    def delete_climb(self, wall_id: str, climb_id: str) -> bool:
        """
        Delete a climb.
        
        Args:
            wall_id: The wall ID
            climb_id: The climb ID
            
        Returns:
            True if deleted, False if not found
        """
        with get_db() as conn:
            cursor = conn.execute(
                "DELETE FROM climbs WHERE id = ? AND wall_id = ?",
                (climb_id, wall_id),
            )
        return cursor.rowcount > 0
    
    def get_climbs_for_training(self, wall_id: str) -> list[dict]:
        """
        Get all climbs for a wall in format suitable for ML training.
        
        Args:
            wall_id: The wall ID
            
        Returns:
            List of climb dicts with parsed sequences
        """
        with get_db() as conn:
            rows = conn.execute(
                "SELECT id, sequence FROM climbs WHERE wall_id = ?",
                (wall_id,),
            ).fetchall()
        
        return [
            {"id": row["id"], "sequence": json.loads(row["sequence"])}
            for row in rows
        ]
    
    def _row_to_climb(self, row) -> Climb:
        """Convert a database row to a Climb object."""
        sequence = json.loads(row["sequence"])
        return Climb(
            id=row["id"],
            wall_id=row["wall_id"],
            name=row["name"],
            grade=row["grade"],
            setter=row["setter"],
            sequence=sequence,
            tags=json.loads(row["tags"]) if row["tags"] else None,
            num_moves=len(sequence),
            created_at=row["created_at"],
        )
