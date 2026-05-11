"""
Service for managing climbs.
"""
import json
import uuid
from datetime import datetime

from app.database import get_db
from app.schemas import Climb, ClimbCreate, ClimbSortBy, Holdset
from app.services.utils import _get_layout_angle


def get_climbs(
    layout_id: str,
    min_difficulty: float,
    max_difficulty: float,
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
    conditions = ["(layout_id = ? OR layout_id = ?)"]
    params: list = [layout_id, layout_id]

    if angle is not None:
        conditions.append("angle = ?")
        params.append(angle)

    if include_projects:
        conditions.append("(difficulty IS NULL OR (difficulty >= ? AND difficulty <= ?))")
    else:
        conditions.append("(difficulty IS NOT NULL AND difficulty >= ? AND difficulty <= ?)")
    params.extend([min_difficulty, max_difficulty])

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
        total = conn.execute(
            f"SELECT COUNT(*) FROM climbs WHERE {where_clause}", params
        ).fetchone()[0]

        rows = conn.execute(
            f"""
            SELECT * FROM climbs
            WHERE {where_clause}
            ORDER BY {sort_column} {order_direction}
            LIMIT ? OFFSET ?
            """,
            params + [limit, offset],
        ).fetchall()

    return [_row_to_climb(row) for row in rows], total, limit, offset


def create_climb(layout_id: str, climb_data: ClimbCreate) -> str:
    climb_id = f"climb-{uuid.uuid4().hex[:15]}"
    angle = climb_data.angle if climb_data.angle else _get_layout_angle(layout_id)
    hold_list = _holdset_to_holds(climb_data.holdset)

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
                 difficulty, quality, ascents, setter_name, setter_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                climb_id,
                layout_id,
                layout_id,
                angle,
                climb_data.name,
                json.dumps(hold_list),
                json.dumps(climb_data.tags) if climb_data.tags else None,
                climb_data.difficulty,
                climb_data.quality,
                climb_data.ascents,
                climb_data.setter_name,
                climb_data.setter_id,
            ),
        )
    return climb_id


def create_climbs_batch(layout_id: str, climbs_data: list[ClimbCreate]) -> list[dict]:
    results = []
    with get_db() as conn:
        for index, climb_data in enumerate(climbs_data):
            try:
                climb_id = f"climb-{uuid.uuid4().hex[:12]}"
                conn.execute(
                    """
                    INSERT INTO climbs
                        (id, layout_id, layout_id, angle, name, holds, tags,
                         difficulty, quality, ascents, setter_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        climb_id,
                        layout_id,
                        layout_id,
                        climb_data.angle,
                        climb_data.name,
                        json.dumps(_holdset_to_holds(climb_data.holdset)),
                        json.dumps(climb_data.tags) if climb_data.tags else None,
                        climb_data.difficulty,
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
    with get_db() as conn:
        row = conn.execute(
            "SELECT setter_id FROM climbs WHERE id = ? AND (layout_id = ? OR layout_id = ?)",
            (climb_id, layout_id, layout_id),
        ).fetchone()
    return row["setter_id"] if row else None


def delete_climb(layout_id: str, climb_id: str) -> bool:
    with get_db() as conn:
        cursor = conn.execute(
            "DELETE FROM climbs WHERE id = ? AND (layout_id = ? OR layout_id = ?)",
            (climb_id, layout_id, layout_id),
        )
    return cursor.rowcount > 0


def get_climbs_for_training(layout_id: str, tags: list[str] | None = None) -> list[dict]:
    conditions = ["(layout_id = ? OR layout_id = ?)"]
    params: list = [layout_id, layout_id]
    if tags:
        for tag in tags:
            conditions.append("""
                EXISTS (SELECT 1 FROM json_each(tags) WHERE value = ?)
            """)
            params.append(tag)
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT id, holds FROM climbs WHERE {' AND '.join(conditions)}",
            params,
        ).fetchall()
    return [{"id": row["id"], "holds": json.loads(row["holds"])} for row in rows]


def _row_to_climb(row) -> Climb:
    holds = json.loads(row["holds"])
    holdset = _holds_to_holdset(holds)
    layout_id = row["layout_id"] or row["layout_id"]
    return Climb(
        id=row["id"],
        layout_id=layout_id,
        angle=row["angle"],
        name=row["name"],
        holdset=holdset,
        difficulty=row["difficulty"] if row["difficulty"] else None,
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
        case ClimbSortBy.DIFFICULTY:
            return "difficulty"
        case ClimbSortBy.ASCENTS:
            return "ascents"
        case _:
            return "created_at"


def _holds_to_holdset(holds: list[list[int]]) -> Holdset:
    roles: list[list[int]] = [[], [], [], []]
    for h in holds:
        roles[h[1]].append(h[0])
    return Holdset(start=roles[0], finish=roles[1], hand=roles[2], foot=roles[3])


def _holdset_to_holds(holdset: Holdset) -> list[list[int]]:
    holds = []
    for role, hold_list in enumerate([holdset.start, holdset.finish, holdset.hand, holdset.foot]):
        holds.extend([[h_idx, role] for h_idx in hold_list])
    return holds
