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
from app.schemas import Climb, ClimbCreate, ClimbSortBy, Holdset



def get_climbs(
    wall_id: str,
    angle: int | None = None,
    grade_range: list[int] = [0,39],
    include_projects: bool = True,
    setter_name: str | None = None,        
    name_includes: str | None = None,
    holds_include: list[int] | None = None,
    tags_include: list[str] | None = None,
    after: datetime | None = None,
    sort_by: ClimbSortBy = ClimbSortBy.DATE,
    descending: bool = True,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[Climb], int, int, int]:
    """
    Get climbs for a wall with filtering.
    
    Args:
        wall_id: The wall ID
        angle: Filter by wall angle (optional)
        grade_range: Range of grade (converted to decimal V-grade; v9 = 90, v3- = 27)
        include_projects: Whether to include ungraded climbs
        setter_name: Filter by setter name
        name_includes: Filter by name (partial match)
        holds_include: Hold indices that must be in the climb
        tags_include: Tags that the climb must have
        after: Filter climbs created after this date
        sort_by: Sort order
        descending: whether to order descending
        limit: Maximum results
        offset: Pagination offset
        
    Returns:
        Tuple of (list of climbs, total count, limit, offset)
    """
    # Build WHERE clauses
    conditions = ["wall_id = ?"]
    params: list = [wall_id]
    
    if angle is not None:
        conditions.append("angle = ?")
        params.append(angle)
    
    if include_projects:
        # Projects (NULL) OR within grade range
        conditions.append("(grade IS NULL OR (grade >= ? AND grade <= ?))")
    else:
        # Only graded climbs within range
        conditions.append("(grade IS NOT NULL AND grade >= ? AND grade <= ?)")
    params.extend([grade_range[0], grade_range[1]])
    
    if setter_name:
        conditions.append("setter_name = ?")
        params.append(setter_name)
    
    if name_includes:
        conditions.append("name LIKE ?")
        params.append(f"%{name_includes}%")
    
    if after:
        conditions.append("created_at > ?")
        params.append(after.isoformat())
    
    
    # Filter by holds - check if hold index appears in holds list
    # Holds format is [[hold_idx, role], [hold_idx, role], ...]
    # We check if hold_idx matches the first element of any sublist
    if holds_include:
        for hold_id in holds_include:
            conditions.append("""
                EXISTS (
                    SELECT 1 FROM json_each(holds) AS hold
                    WHERE json_extract(hold.value, '$[0]') = ?
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
    sort_column = _get_sort_column(sort_by)
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
    
    climbs = [_row_to_climb(row) for row in rows]
    return climbs, total, limit, offset

def create_climb(wall_id: str, climb_data: ClimbCreate) -> str:
    """
    Create a new climb.
    
    Args:
        wall_id: The wall ID
        climb_data: Climb data
        
    Returns:
        The new climb ID
    """
    id = f"climb-{uuid.uuid4().hex[:12]}"
    hold_list = _holdset_to_holds(climb_data.holdset)
    holds = json.dumps(hold_list)
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO climbs (id, wall_id, angle, name, holds, tags, grade, quality, ascents, setter_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                id,
                wall_id,
                climb_data.angle,
                climb_data.name,
                holds,
                json.dumps(climb_data.tags) if climb_data.tags else None,
                climb_data.grade,
                climb_data.quality,
                climb_data.ascents,
                climb_data.setter_name,
            ),
        )
    
    return id

def create_climbs_batch(wall_id: str, climbs_data: list[ClimbCreate]) -> list[dict]:
    """
    Create multiple climbs in a single transaction.
    
    Args:
        wall_id: The wall ID
        climbs_data: List of climb data
        
    Returns:
        List of results with index, id, status, and error (if any)
    """
    results = []
    
    with get_db() as conn:
        for index, climb_data in enumerate(climbs_data):
            try:
                climb_id = f"climb-{uuid.uuid4().hex[:12]}"
                hold_list = _holdset_to_holds(climb_data.holdset)
                holds = json.dumps(hold_list)
                
                conn.execute(
                    """
                    INSERT INTO climbs (id, wall_id, angle, name, holds, tags, grade, quality, ascents, setter_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        climb_id,
                        wall_id,
                        climb_data.angle,
                        climb_data.name,
                        holds,
                        json.dumps(climb_data.tags) if climb_data.tags else None,
                        climb_data.grade,
                        climb_data.quality,
                        climb_data.ascents,
                        climb_data.setter_name,
                    ),
                )
                
                results.append({
                    'index': index,
                    'id': climb_id,
                    'status': 'success',
                    'error': None
                })
                
            except Exception as e:
                results.append({
                    'index': index,
                    'id': None,
                    'status': 'error',
                    'error': str(e)
                })
    
    return results

def delete_climb(wall_id: str, climb_id: str) -> bool:
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
    wall_id: str,
    tags: list[str] | None = None,
) -> list[dict]:
    """
    Get all climbs for a wall in format suitable for ML training.
    
    Args:
        wall_id: The wall ID
        tags: Optional list of tags - climb must have ALL specified tags
        
    Returns:
        List of climb dicts with parsed holds
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
            f"SELECT id, holds FROM climbs WHERE {where_clause}",
            params,
        ).fetchall()
    
    return [
        {"id": row["id"], "holds": json.loads(row["holds"])}
        for row in rows
    ]

def _row_to_climb(row) -> Climb:
    """Convert a database row to a Climb object."""
    holds = json.loads(row["holds"])
    holdset = _holds_to_holdset(holds)
    return Climb(
        id=row["id"],
        wall_id=row["wall_id"],
        angle=row["angle"],
        name=row["name"],
        holdset=holdset,
        grade=row["grade"] if row["grade"] else None,
        quality=row["quality"] if row["quality"] else None,
        ascents=row["ascents"],
        setter_name=row["setter_name"] if row["setter_name"] else None,
        tags=json.loads(row["tags"]) if row["tags"] else None,
        created_at=row["created_at"],
    )

def _get_sort_column(sort_by: ClimbSortBy) -> str:
    """Map sort enum to SQL column/expression."""
    match sort_by:
        case ClimbSortBy.DATE:
            return "created_at"
        case ClimbSortBy.NAME:
            return "name"
        case ClimbSortBy.GRADE:
            return "grade"
        case ClimbSortBy.ASCENTS:
            return "ascents"
        case _:
            return "created_at"

def _holds_to_holdset(holds: list[list[int,int]]):
    """Converts a list of [hold_idx, role] back to a Holdset object."""
    roles = [[],[],[],[]]
    for h in holds:
        roles[h[1]].append(h[0])
    return Holdset(
        start=roles[0],
        finish=roles[1],
        hand=roles[2],
        foot=roles[3],
    )

def _holdset_to_holds(holdset: Holdset) -> list[list[int,int]]:
    """Converts a Holdset object into a list of [idx, role] for each hold within the holdset."""
    holds = []
    for role, hold_list in enumerate([holdset.start, holdset.finish, holdset.hand, holdset.foot]):
        holds.extend([[h_idx,role] for h_idx in hold_list])
    return holds