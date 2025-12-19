"""
Service for managing climbs.

Handles:
- Climb CRUD operations
- Filtering and sorting queries
"""
import json
import uuid
from datetime import datetime

from app.database import get_db
from app.schemas import Climb, ClimbCreate, ClimbSortBy


class ClimbService:
    """Service for climb operations."""
    
    def get_climbs(
        self,
        wall_id: str,
        grade_range: str | None = None,
        include_projects: bool = True,
        setter: str | None = None,        
        name_includes: str | None = None,
        holds_include: list[int] | None = None,
        tags_include: list[str] | None = None,
        after: datetime | None = None,
        sort_by: ClimbSortBy = ClimbSortBy.DATE,
        descending: bool = True,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Climb], int]:
        """
        Get climbs for a wall with filtering.
        
        Args:
            wall_id: The wall ID
            setter: Filter by setter ID
            name_includes: Filter by name (partial match)
            holds_include: Hold IDs that must be in the climb
            tags_include: Tags that the climb must have
            after: Filter climbs created after this date
            sort_by: Sort order
            descending: whether to order descending
            limit: Maximum results
            offset: Pagination offset
            
        Returns:
            Tuple of (list of climbs, total count)
        """
        # Build WHERE clauses
        conditions = ["wall_id = ?"]
        params: list = [wall_id]
        
        if setter:
            conditions.append("setter = ?")
            params.append(setter)
        
        if name_includes:
            conditions.append("name LIKE ?")
            params.append(f"%{name_includes}%")
        
        if after:
            conditions.append("created_at > ?")
            params.append(after.isoformat())
        
        # Filter by holds - check if hold ID appears anywhere in sequence
        # Sequence is [[lh, rh], [lh, rh], ...] so we need nested json_each
        if holds_include:
            for hold_id in holds_include:
                conditions.append("""
                    EXISTS (
                        SELECT 1 FROM json_each(sequence) AS pos
                        WHERE EXISTS (
                            SELECT 1 FROM json_each(pos.value) AS hold
                            WHERE hold.value = ?
                        )
                    )
                """)
                params.append(hold_id)
        
        # Filter by tags - check if tag exists in tags array
        if tags_include:
            for tag in tags_include:
                conditions.append("""
                    EXISTS (
                        SELECT 1 FROM json_each(tags) 
                        WHERE value = ?
                    )
                """)
                params.append(tag)
        
        where_clause = " AND ".join(conditions)
        
        # Build ORDER BY clause
        sort_column = self._get_sort_column(sort_by)
        order_direction = "DESC" if descending else "ASC"
        order_clause = f"{sort_column} {order_direction}"
        
        with get_db() as conn:
            # Get total count (without pagination)
            count_query = f"SELECT COUNT(*) FROM climbs WHERE {where_clause}"
            total = conn.execute(count_query, params).fetchone()[0]
            
            # Get paginated results
            query = f"""
                SELECT * FROM climbs 
                WHERE {where_clause}
                ORDER BY {order_clause}
                LIMIT ? OFFSET ?
            """
            rows = conn.execute(query, params + [limit, offset]).fetchall()
        
        climbs = [self._row_to_climb(row) for row in rows]
        return climbs, total
    
    def _get_sort_column(self, sort_by: ClimbSortBy) -> str:
        """Map sort enum to SQL column/expression."""
        match sort_by:
            case ClimbSortBy.DATE:
                return "created_at"
            case ClimbSortBy.NAME:
                return "name"
            case ClimbSortBy.GRADE:
                return "grade"
            case ClimbSortBy.TICKS:
                # Ticks not yet implemented - fall back to date
                return "created_at"
            case ClimbSortBy.NUM_MOVES:
                return "json_array_length(sequence)"
            case _:
                return "created_at"
    
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
    
    def get_climbs_for_training(
        self, 
        wall_id: str, 
        tags: list[str] | None = None,
    ) -> list[dict]:
        """
        Get all climbs for a wall in format suitable for ML training.
        
        Args:
            wall_id: The wall ID
            tags: Optional list of tags - climb must have ALL specified tags
            
        Returns:
            List of climb dicts with parsed sequences
        """
        conditions = ["wall_id = ?"]
        params: list = [wall_id]
        
        # Filter by tags if provided
        if tags:
            for tag in tags:
                conditions.append("""
                    EXISTS (
                        SELECT 1 FROM json_each(tags) 
                        WHERE value = ?
                    )
                """)
                params.append(tag)
        
        where_clause = " AND ".join(conditions)
        
        with get_db() as conn:
            rows = conn.execute(
                f"SELECT id, sequence FROM climbs WHERE {where_clause}",
                params,
            ).fetchall()
        
        return [
            {"id": row["id"], "sequence": json.loads(row["sequence"])}
            for row in rows
        ]
    
    def _row_to_climb(self, row) -> Climb:
        """Convert a database row to a Climb object."""
        sequence = json.loads(row["sequence"]) if row["sequence"] else []
        return Climb(
            id=row["id"],
            wall_id=row["wall_id"],
            name=row["name"],
            grade=row["grade"],
            setter=row["setter"],
            sequence=sequence,
            tags=json.loads(row["tags"]) if row["tags"] else None,
            num_moves=len(sequence) - 1 if sequence else 0,
            created_at=row["created_at"],
        )