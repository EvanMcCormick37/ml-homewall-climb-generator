"""
Database setup and connection management.
Uses SQLite for simplicity - stores climbs and job queue.
"""
import sqlite3
from contextlib import contextmanager
from typing import Generator
from app.config import settings


def get_db_connection() -> sqlite3.Connection:
    """Get a database connection with row factory."""
    conn = sqlite3.connect(settings.DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """Context manager for database connections."""
    conn = get_db_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Return True if a table exists in the database."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def _column_exists(conn: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    """Return True if a column exists in a table."""
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(row["name"] == column_name for row in rows)


def _add_column_if_missing(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    """Add a column to a table only if it does not already exist."""
    if not _column_exists(conn, table, column):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def init_db():
    """Initialize database tables and run any pending migrations."""
    settings.DATA_DIR.mkdir(exist_ok=True)
    settings.WALLS_DIR.mkdir(exist_ok=True)
    settings.LAYOUTS_DIR.mkdir(exist_ok=True)

    with get_db() as conn:
        cursor = conn.cursor()

        # ── Users ─────────────────────────────────────────────────────────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL,
                display_name TEXT,
                avatar_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ── Layouts ───────────────────────────────────────────────────────────
        # A layout is a unique hold arrangement (was previously "wall").
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS layouts (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                num_holds INTEGER DEFAULT 0,
                owner_id TEXT,
                visibility TEXT DEFAULT 'public',
                share_token TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ── Sizes ─────────────────────────────────────────────────────────────
        # A size is a physical variant of a layout (different dimensions/photo).
        # edge_* define which holds from the layout's master set are in-bounds.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sizes (
                id TEXT PRIMARY KEY,
                layout_id TEXT NOT NULL,
                name TEXT NOT NULL,
                width_ft REAL,
                height_ft REAL,
                edge_left REAL NOT NULL DEFAULT 0.0,
                edge_right REAL,
                edge_bottom REAL NOT NULL DEFAULT 0.0,
                edge_top REAL,
                photo_path TEXT,
                num_climbs INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (layout_id) REFERENCES layouts(id) ON DELETE CASCADE
            )
        """)

        # ── Walls (legacy) ───────────────────────────────────────────────────
        # Kept for backward compat until Phase 6 cleanup.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS walls (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                photo_path TEXT NOT NULL,
                num_holds INTEGER DEFAULT 0,
                num_climbs INTEGER DEFAULT 0,
                dimensions TEXT,
                angle INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                owner_id TEXT,
                visibility TEXT DEFAULT 'public',
                share_token TEXT
            )
        """)

        # ── Holds ─────────────────────────────────────────────────────────────
        # Holds belong to a layout (shared across all sizes of that layout).
        # layout_id is the new FK; wall_id is kept for backward compat.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS holds (
                id TEXT PRIMARY KEY,
                wall_id TEXT,
                layout_id TEXT,
                hold_index INTEGER NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                pull_x REAL DEFAULT 0.0,
                pull_y REAL DEFAULT 0.0,
                useability REAL DEFAULT 0.5,
                is_foot INTEGER DEFAULT 0,
                tags TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ── Climbs ────────────────────────────────────────────────────────────
        # Climbs belong to a layout; size_id optionally records what size they
        # were set on. wall_id is kept for backward compat.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS climbs (
                id TEXT PRIMARY KEY,
                wall_id TEXT,
                layout_id TEXT,
                size_id TEXT,
                angle INTEGER NOT NULL,
                name TEXT NOT NULL,
                num_holds INTEGER NOT NULL DEFAULT 0,
                holds TEXT NOT NULL,
                tags TEXT,
                grade REAL,
                quality REAL DEFAULT 2.5,
                ascents INTEGER DEFAULT 0,
                setter_name TEXT,
                setter_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (layout_id) REFERENCES layouts(id) ON DELETE CASCADE,
                FOREIGN KEY (size_id) REFERENCES sizes(id) ON DELETE SET NULL
            )
        """)

        # ── Additive column migrations (idempotent) ───────────────────────────
        # These handle existing databases that were created before this schema.
        _add_column_if_missing(conn, "holds", "layout_id", "TEXT")
        _add_column_if_missing(conn, "holds", "wall_id", "TEXT")
        _add_column_if_missing(conn, "climbs", "layout_id", "TEXT")
        _add_column_if_missing(conn, "climbs", "size_id", "TEXT")
        _add_column_if_missing(conn, "climbs", "wall_id", "TEXT")

        # ── Data migration (walls → layouts + sizes) ──────────────────────────
        _run_migration_if_needed(conn)


def _run_migration_if_needed(conn: sqlite3.Connection) -> None:
    """
    Detect old schema (walls table populated, layouts table empty) and migrate.

    Safe to call on every startup — checks guard conditions before doing work.
    """
    if not _table_exists(conn, "walls"):
        return  # Fresh install with new schema, nothing to migrate

    # Count walls that haven't been migrated yet
    wall_count = conn.execute("SELECT COUNT(*) FROM walls").fetchone()[0]
    layout_count = conn.execute("SELECT COUNT(*) FROM layouts").fetchone()[0]

    if wall_count == 0 or layout_count >= wall_count:
        # Nothing to migrate (already done or no data)
        _backfill_layout_id(conn)
        return

    import uuid as _uuid
    from datetime import datetime

    walls = conn.execute("SELECT * FROM walls").fetchall()
    migrated = 0

    for wall in walls:
        wall_id = wall["id"]

        # Skip if this layout already exists (partial migration resume)
        exists = conn.execute(
            "SELECT 1 FROM layouts WHERE id = ?", (wall_id,)
        ).fetchone()
        if exists:
            continue

        # ── Create layout (same id as wall) ──────────────────────────────────
        conn.execute(
            """
            INSERT INTO layouts (id, name, description, num_holds, owner_id,
                                 visibility, share_token, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                wall_id,
                wall["name"],
                None,
                wall["num_holds"] or 0,
                wall["owner_id"],
                wall["visibility"] or "public",
                wall["share_token"],
                wall["created_at"],
                wall["updated_at"],
            ),
        )

        # ── Parse dimensions from "width, height" string ─────────────────────
        width_ft, height_ft = None, None
        if wall["dimensions"]:
            try:
                parts = wall["dimensions"].split(",")
                width_ft = float(parts[0].strip())
                height_ft = float(parts[1].strip())
            except (ValueError, IndexError):
                pass

        # ── Create the single size for this wall ─────────────────────────────
        size_id = f"size-{_uuid.uuid4().hex[:12]}"
        size_name = (
            f"{int(width_ft)}×{int(height_ft)}"
            if (width_ft and height_ft)
            else "Default"
        )
        conn.execute(
            """
            INSERT INTO sizes (id, layout_id, name, width_ft, height_ft,
                               edge_left, edge_right, edge_bottom, edge_top,
                               photo_path, num_climbs, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                size_id,
                wall_id,
                size_name,
                width_ft,
                height_ft,
                0.0,
                width_ft,
                0.0,
                height_ft,
                wall["photo_path"],  # filename only, resolves via WALLS_DIR/wall_id/
                0,
                wall["created_at"],
                wall["updated_at"],
            ),
        )

        migrated += 1

    if migrated:
        print(f"[migration] Migrated {migrated} wall(s) → layouts + sizes")

    # Backfill layout_id in holds and climbs from wall_id
    _backfill_layout_id(conn)


def _backfill_layout_id(conn: sqlite3.Connection) -> None:
    """
    Copy wall_id → layout_id in holds and climbs rows where layout_id is NULL.
    Safe to call multiple times (only touches NULL rows).
    """
    if _column_exists(conn, "holds", "wall_id") and _column_exists(conn, "holds", "layout_id"):
        conn.execute("""
            UPDATE holds SET layout_id = wall_id
            WHERE layout_id IS NULL AND wall_id IS NOT NULL
        """)

    if _column_exists(conn, "climbs", "wall_id") and _column_exists(conn, "climbs", "layout_id"):
        conn.execute("""
            UPDATE climbs SET layout_id = wall_id
            WHERE layout_id IS NULL AND wall_id IS NOT NULL
        """)
