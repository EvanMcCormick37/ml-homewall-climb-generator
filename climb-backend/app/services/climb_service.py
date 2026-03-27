"""
Service for managing climbs.

Handles:
- Climb CRUD operations (keyed by layout_id for new code)
- Filtering and sorting queries
- Legacy layout_id support (backward compat until Phase 6 cleanup)
"""
import json
import uuid
from datetime import datetime

from app.database import get_db
from app.schemas import Climb, ClimbCreate, ClimbSortBy, Holdset
from app.services.utils import GRADE_TO_DIFF, _get_layout_angle


def _grade_range(min_grade: str, max_grade: str, grade_scale: str):
    """Compute the difficulty range from min/max grade strings."""
    min_d = GRADE_TO_DIFF[grade_scale][min_grade]
    max_d = GRADE_TO_DIFF[grade_scale][max_grade]
    return min_d, max_d


def get_climbs(
    layout_id: str,
    grade_scale: str,
    min_grade: str,
    max_grade: str,
    angle: int | None = None,
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
    size_id: str | None = None,
) -> tuple[list[Climb], int, int, int]:
    """
    Get climbs for a layout/layout with filtering.

    `layout_id` is accepted as the primary lookup key; it matches both the legacy
    `layout_id` column and the new `layout_id` column (they are equal for all
    migrated data). `size_id` is accepted for future size-aware filtering but
    is not yet applied as a hard filter (climbs are not yet tagged with size_id
    in the existing dataset).
    """
    grade_range = _grade_range(min_grade, max_grade, grade_scale)

    # Match on layout_id OR layout_id so both old and new data is returned
    conditions = ["(layout_id = ? OR layout_id = ?)"]
    params: list = [layout_id, layout_id]

    if angle is not None:
        conditions.append("angle = ?")
        params.append(angle)

    if include_projects:
        conditions.append("(grade IS NULL OR (grade >= ? AND grade <= ?))")
    else:
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

    if holds_include:
        for hold_id in holds_include:
            conditions.append("""
                EXISTS (
                    SELECT 1 FROM json_each(holds) AS hold
                    WHERE json_extract(hold.value, '$[0]') = ?
                )
            """)
            params.append(hold_id)

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
    sort_column = _get_sort_column(sort_by)
    order_direction = "DESC" if descending else "ASC"

    with get_db() as conn:
        count_query = f"SELECT COUNT(*) FROM climbs WHERE {where_clause}"
        total = conn.execute(count_query, params).fetchone()[0]

        query = f"""
            SELECT * FROM climbs
            WHERE {where_clause}
            ORDER BY {sort_column} {order_direction}
            LIMIT ? OFFSET ?
        """
        rows = conn.execute(query, params + [limit, offset]).fetchall()

    climbs = [_row_to_climb(row) for row in rows]
    return climbs, total, limit, offset


def create_climb(layout_id: str, climb_data: ClimbCreate) -> str:
    """
    Create a new climb.

    `layout_id` is accepted for backward compat with the legacy router; it is
    stored in both `layout_id` and `layout_id` columns.
    """
    climb_id = f"climb-{uuid.uuid4().hex[:15]}"
    angle = climb_data.angle if climb_data.angle else _get_layout_angle(layout_id)
    grade = (
        GRADE_TO_DIFF[climb_data.scale][climb_data.grade]
        if (climb_data.scale and climb_data.grade)
        else None
    )
    hold_list = _holdset_to_holds(climb_data.holdset)
    holds = json.dumps(hold_list)

    with get_db() as conn:
        duplicate = conn.execute(
            """
            SELECT id FROM climbs
            WHERE (layout_id = ? OR layout_id = ?) AND name = ? AND angle = ? AND setter_id = ?
            """,
            (layout_id, layout_id, climb_data.name, angle, climb_data.setter_id),
        ).fetchone()
        if duplicate:
            raise ValueError(
                f"A climb with the same name, angle, and setter already exists (id: {duplicate['id']})"
            )
        conn.execute(
            """
            INSERT INTO climbs
                (id, layout_id, layout_id, angle, name, holds, tags,
                 grade, quality, ascents, setter_name, setter_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                climb_id,
                layout_id,    # legacy column
                layout_id,    # new column (same value for now)
                angle,
                climb_data.name,
                holds,
                json.dumps(climb_data.tags) if climb_data.tags else None,
                grade,
                climb_data.quality,
                climb_data.ascents,
                climb_data.setter_name,
                climb_data.setter_id,
            ),
        )
    return climb_id


def create_climbs_batch(layout_id: str, climbs_data: list[ClimbCreate]) -> list[dict]:
    """Create multiple climbs in a single transaction."""
    results = []

    with get_db() as conn:
        for index, climb_data in enumerate(climbs_data):
            try:
                climb_id = f"climb-{uuid.uuid4().hex[:12]}"
                hold_list = _holdset_to_holds(climb_data.holdset)
                holds = json.dumps(hold_list)

                conn.execute(
                    """
                    INSERT INTO climbs
                        (id, layout_id, layout_id, angle, name, holds, tags,
                         grade, quality, ascents, setter_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        climb_id,
                        layout_id,
                        layout_id,
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
                results.append({"index": index, "id": climb_id, "status": "success", "error": None})

            except Exception as e:
                results.append({"index": index, "id": None, "status": "error", "error": str(e)})

    return results


def get_climb_setter_id(layout_id: str, climb_id: str) -> str | None:
    """Return the setter_id of a climb, or None if not found."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT setter_id FROM climbs WHERE id = ? AND (layout_id = ? OR layout_id = ?)",
            (climb_id, layout_id, layout_id),
        ).fetchone()
    return row["setter_id"] if row else None


def delete_climb(layout_id: str, climb_id: str) -> bool:
    """Delete a climb."""
    with get_db() as conn:
        cursor = conn.execute(
            "DELETE FROM climbs WHERE id = ? AND (layout_id = ? OR layout_id = ?)",
            (climb_id, layout_id, layout_id),
        )
    return cursor.rowcount > 0


def get_climbs_for_training(
    layout_id: str,
    tags: list[str] | None = None,
) -> list[dict]:
    """Get all climbs for a layout/layout in format suitable for ML training."""
    conditions = ["(layout_id = ? OR layout_id = ?)"]
    params: list = [layout_id, layout_id]

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

    return [{"id": row["id"], "holds": json.loads(row["holds"])} for row in rows]


def _row_to_climb(row) -> Climb:
    """Convert a database row to a Climb object."""
    holds = json.loads(row["holds"])
    holdset = _holds_to_holdset(holds)
    # Prefer layout_id; fall back to layout_id for any pre-migration rows
    layout_id = row["layout_id"] or row["layout_id"]
    return Climb(
        id=row["id"],
        layout_id=layout_id,
        angle=row["angle"],
        name=row["name"],
        holdset=holdset,
        grade=row["grade"] if row["grade"] else None,
        quality=row["quality"] if row["quality"] else None,
        ascents=row["ascents"],
        setter_name=row["setter_name"] if row["setter_name"] else None,
        setter_id=row["setter_id"] if row["setter_id"] else None,
        tags=json.loads(row["tags"]) if row["tags"] else None,
        created_at=row["created_at"],
    )


def _get_sort_column(sort_by: ClimbSortBy) -> str:
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


def _holds_to_holdset(holds: list[list[int]]) -> Holdset:
    """Convert a list of [hold_idx, role] back to a Holdset object."""
    roles: list[list[int]] = [[], [], [], []]
    for h in holds:
        roles[h[1]].append(h[0])
    return Holdset(
        start=roles[0],
        finish=roles[1],
        hand=roles[2],
        foot=roles[3],
    )


def _holdset_to_holds(holdset: Holdset) -> list[list[int]]:
    """Convert a Holdset into a list of [idx, role] pairs."""
    holds = []
    for role, hold_list in enumerate([holdset.start, holdset.finish, holdset.hand, holdset.foot]):
        holds.extend([[h_idx, role] for h_idx in hold_list])
    return holds
