"""
db_migration.py

Utilities for migrating climbing data from boardlib SQLite databases into
data/storage.db for model training.

Supported source databases (located in data/boardlib/):
    kilter, tension, grasshopper, decoy

The destination (data/storage.db) mirrors the application backend schema but is
used exclusively for training — it does NOT need to satisfy the app's relational
constraints (e.g. migrated climbs' wall_id values have no matching row in walls).

Role integer convention (matches storage.db / climb_conversion.py):
    0 = start
    1 = finish
    2 = middle / hand
    3 = foot
"""

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

BOARDLIB_DIR = Path("data/boardlib")
STORAGE_DB   = Path("data/storage.db")

SOURCE_DBS_WITH_LAYOUTS = {
    "kilter":[1,8],
    "tension":[9, 10, 11],
    "grasshopper":[1],
    "decoy":[2]
}

# Maps boardlib placement_role.name → storage.db role integer
ROLE_NAME_TO_INT: dict[str, int] = {
    "start":  0,
    "finish": 1,
    "middle": 2,
    "foot":   3,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_role_map(src_conn: sqlite3.Connection) -> dict[int, int]:
    """
    Build a mapping from placement_role.id (int) to storage.db role integer
    by reading the source database's placement_roles table.

    boardlib stores role names ('start', 'finish', 'middle', 'foot') in
    placement_roles.name; the IDs vary across board types (e.g. kilter uses
    IDs 12-15, 20-23, …, while tension/grasshopper/decoy use 1-4).

    Args:
        src_conn: Open connection to a boardlib source database.

    Returns:
        Dict mapping each known role ID to its storage.db integer (0-3).
        IDs whose name is not in ROLE_NAME_TO_INT (e.g. kilter's 'cyan',
        'magenta' coloured-hold roles) are omitted — holds with those role
        IDs will be dropped during frame parsing.
    """
    rows = src_conn.execute("SELECT id, name FROM placement_roles").fetchall()
    role_map: dict[int, int] = {}
    for role_id, name in rows:
        role_int = ROLE_NAME_TO_INT.get(name)
        if role_int is not None:
            role_map[role_id] = role_int
    return role_map


def _parse_frames(frames: str, role_map: dict[int, int]) -> list[list[int]]:
    """
    Parse a boardlib frames string into a list of [hold_index, role_int] pairs.

    The frames format is a concatenation of tokens: p{hole_id}r{role_id}
    e.g. "p1100r12p1200r13p1350r14p1400r15"

    Args:
        frames:   Raw frames string from boardlib climbs.frames.
        role_map: Mapping from boardlib role ID to storage.db role integer,
                  as returned by _build_role_map().

    Returns:
        List of [hold_index, role_int] pairs in parse order.
        Holds whose role_id is absent from role_map are silently dropped
        (this covers kilter's decorative colour roles).
        Returns an empty list if the string is unparseable or all roles unknown.
    """
    holds: list[list[int]] = []
    for p_str, r_str in re.findall(r'p(\d+)r(\d+)', frames):
        role_int = role_map.get(int(r_str))
        if role_int is not None:
            holds.append([int(p_str), role_int])
    return holds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def migrate_boardlib_climbs(
    source_db_names: dict[str,list[int]] = SOURCE_DBS_WITH_LAYOUTS,
    boardlib_dir: str | Path    = BOARDLIB_DIR,
    storage_db:   str | Path    = STORAGE_DB,
    min_ascents:  int           = 0,
    skip_existing: bool         = False,
    verbose:      bool          = True,
) -> dict[str, int]:
    """
    Migrate climbs from boardlib SQLite databases into data/storage.db.

    For each source database, every row in climb_stats is treated as one
    training example — this naturally handles boards (kilter, tension) where
    a single climb can be set at multiple angles, each with independent grade
    and quality statistics.  The join is:

        climb_stats  INNER JOIN  climbs  ON  uuid = climb_uuid

    so only climbs that have at least one stats row are migrated.  Climbs
    without stats carry no grade or quality signal and are not useful for
    conditional generation training.

    Schema mapping (boardlib → storage.db climbs):
        id          ← "boardlib-{db_name}-{uuid}-{angle}"  (synthetic, unique)
        wall_id     ← db_name  (e.g. "kilter"; no matching row in walls table)
        angle       ← climb_stats.angle
        name        ← climbs.uuid  (used as a stable source identifier)
        holds       ← JSON [[hold_index, role_int], ...]  parsed from climbs.frames
        tags        ← NULL
        grade       ← climb_stats.difficulty_average
        quality     ← climb_stats.quality_average
        ascents     ← climb_stats.ascensionist_count
        setter_name ← climbs.setter_username
        created_at  ← climbs.created_at  (falls back to UTC now if NULL)
        setter_id   ← NULL

    Args:
        source_db_names:
            List of boardlib database names to process, without the .db
            extension.  Defaults to ["kilter", "tension", "grasshopper", "decoy"].
        boardlib_dir:
            Directory containing the boardlib .db files.
        storage_db:
            Path to the destination storage.db file.
        min_ascents:
            Skip climb-angle pairs with fewer than this many ascents.
            Defaults to 0 (keep everything).  A value of ~10 is a reasonable
            filter to remove rarely-climbed and potentially mislabelled routes.
        skip_existing:
            If True (default), check for rows already present in storage.db
            (matched by the synthetic id column) and skip them.  Safe to re-run
            incrementally after a partial migration.
        verbose:
            Print per-database progress updates to stdout.

    Returns:
        Dict of {db_name: rows_inserted} for each processed source database.
    """
    boardlib_dir = Path(boardlib_dir)
    storage_db   = Path(storage_db)

    summary: dict[str, int] = {}

    with sqlite3.connect(storage_db) as dest:

        # Empty out the current climb directory
        dest.execute("DELETE FROM climbs")

        for db_name, layouts in source_db_names.items():
            src_path = boardlib_dir / f"{db_name}.db"

            if not src_path.exists():
                if verbose:
                    print(f"[{db_name}] {src_path} not found — skipping.")
                summary[db_name] = 0
                continue

            if verbose:
                print(f"\n[{db_name}] Opening {src_path} ...")

            inserted = 0
            skipped  = 0

            for layout in layouts:
                with sqlite3.connect(src_path) as src:
                    src.row_factory = sqlite3.Row

                    # Build role_id → role_int map for this board type
                    role_map = _build_role_map(src)
                    if verbose:
                        print(f"[{db_name}] Role map: {role_map}")

                    # Load already-present ids to support incremental re-runs
                    if skip_existing:
                        existing_ids: set[str] = {
                            row[0]
                            for row in dest.execute(
                                "SELECT id FROM climbs WHERE layout_id = ?", (f"{db_name}-{layout}",)
                            )
                        }
                    else:
                        existing_ids = set()

                    # Drive the query from climb_stats so each (climb, angle) pair
                    # becomes one row.  This correctly handles multi-angle boards.
                    min_ascents_clause = (
                        f"AND cs.ascensionist_count >= {int(min_ascents)}"
                        if min_ascents > 0 else ""
                    )
                    query = f"""
                        SELECT
                            c.uuid            AS uuid,
                            c.setter_username AS setter_name,
                            c.frames          AS frames,
                            c.created_at      AS created_at,
                            cs.angle          AS angle,
                            cs.difficulty_average  AS grade,
                            cs.quality_average     AS quality,
                            cs.ascensionist_count  AS ascents
                        FROM climb_stats cs
                        INNER JOIN climbs c ON c.uuid = cs.climb_uuid
                        WHERE c.layout_id = ?
                            AND c.is_listed = 1
                            AND c.is_draft  = 0
                            AND c.frames IS NOT NULL
                            AND c.frames   != ''
                        {min_ascents_clause}
                    """

                    rows = src.execute(query, (layout,)).fetchall()

                if verbose:
                    print(f"[{db_name}, layout id {layout}] {len(rows)} candidate (climb, angle) pairs.")

                batch: list[tuple] = []

                for row in rows:
                    climb_id = f"boardlib-{db_name}-{layout}-{row['uuid']}-{row['angle']}"

                    if climb_id in existing_ids:
                        skipped += 1
                        continue

                    holds = _parse_frames(row['frames'], role_map)
                    if not holds:
                        # All frame tokens had unknown role IDs (e.g. purely decorative)
                        skipped += 1
                        continue

                    created_at = row['created_at'] or datetime.now(timezone.utc).isoformat()

                    batch.append((
                        climb_id,               # id
                        f"{db_name}-{layout}",  # layout_id
                        row['angle'],           # angle
                        row['uuid'],            # name  (source identifier)
                        json.dumps(holds),      # holds
                        None,                   # tags
                        row['grade'],           # grade
                        row['quality'],         # quality
                        row['ascents'],         # ascents
                        row['setter_name'],     # setter_name
                        created_at,             # created_at
                        None,                   # setter_id
                    ))

                    # Flush in batches of 1 000 to keep memory usage flat
                    if len(batch) == 1_000:
                        dest.executemany(
                            """
                            INSERT OR IGNORE INTO climbs
                                (id, wall_id, angle, name, holds, tags,
                                grade, quality, ascents, setter_name,
                                created_at, setter_id)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                            """,
                            batch,
                        )
                        inserted += len(batch)
                        batch = []
                        if verbose:
                            print(f"[{db_name}]   {inserted} inserted so far ...")

                # Final partial batch
                if batch:
                    dest.executemany(
                        """
                        INSERT OR IGNORE INTO climbs
                            (id, layout_id, angle, name, holds, tags,
                            grade, quality, ascents, setter_name,
                            created_at, setter_id)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        batch,
                    )
                    inserted += len(batch)

                dest.commit()
                summary[db_name] = inserted

                if verbose:
                    print(
                        f"[{db_name}] Done — {inserted} inserted, {skipped} skipped."
                    )

    if verbose:
        total = sum(summary.values())
        print(f"\nMigration complete. Total rows inserted: {total}")
        for name, count in summary.items():
            print(f"  {name:>12}: {count:>8,} rows")

    return summary
