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
                dimensions INTEGER,
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
                edges TEXT NOT NULL,--serialized list of numbers,
                kickboard BOOLEAN,
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
                layout_id TEXT,
                hold_index INTEGER NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                pull_x REAL DEFAULT 0.0,
                pull_y REAL DEFAULT 0.0,
                useability REAL DEFAULT 0.5,
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
                layout_id TEXT,
                angle INTEGER NOT NULL,
                name TEXT NOT NULL,
                holds TEXT NOT NULL,
                tags TEXT,
                grade REAL,
                quality REAL DEFAULT 2.5,
                ascents INTEGER DEFAULT 0,
                setter_name TEXT,
                setter_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (layout_id) REFERENCES layouts(id) ON DELETE CASCADE,
            )
        """)
