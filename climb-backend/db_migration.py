"""
Standalone database migration script: walls -> layouts + sizes.

Run from the climb-backend/ directory:
    python db_migration.py [--db-path data/storage.db] [--dry-run]

What it does:
  1. For each row in `walls`, creates a matching row in `layouts` (same id).
  2. Creates one row in `sizes` for each wall (using the wall's dimensions/photo).
  3. Adds `layout_id` column to `holds` and `climbs` (if missing).
  4. Backfills `layout_id` from `wall_id` in `holds` and `climbs`.
  5. Does NOT drop the `walls` table (that happens in Phase 6 cleanup).

Safe to re-run — all steps are idempotent.
"""
import argparse
import sqlite3
import uuid
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def table_exists(conn, name):
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def column_exists(conn, table, col):
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r[1] == col for r in rows)


def add_column_if_missing(conn, table, col, defn):
    if not column_exists(conn, table, col):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {defn}")
        print(f"  [schema] Added column {table}.{col}")


def parse_dimensions(dim_str):
    """Parse 'width, height' string -> (float, float) or (None, None)."""
    if not dim_str:
        return None, None
    try:
        parts = dim_str.split(",")
        return float(parts[0].strip()), float(parts[1].strip())
    except (ValueError, IndexError):
        return None, None


# ---------------------------------------------------------------------------
# Migration steps
# ---------------------------------------------------------------------------

def migrate_walls_to_layouts(conn, dry_run=False):
    """Create layout + size rows for every wall that hasn't been migrated yet."""
    if not table_exists(conn, "walls"):
        print("No `walls` table found — skipping (fresh install?).")
        return 0

    walls = conn.execute("SELECT * FROM walls").fetchall()
    if not walls:
        print("No walls to migrate.")
        return 0

    migrated = 0
    skipped = 0

    for wall in walls:
        wall_id = wall[0] if isinstance(wall, tuple) else wall["id"]
        # Access by index if Row factory not set
        try:
            name = wall["name"]
            photo_path = wall["photo_path"]
            num_holds = wall["num_holds"] or 0
            dimensions = wall["dimensions"]
            owner_id = wall["owner_id"]
            visibility = wall["visibility"] or "public"
            share_token = wall["share_token"]
            created_at = wall["created_at"]
            updated_at = wall["updated_at"]
        except (TypeError, IndexError):
            # Fallback tuple access (col order: id, name, photo_path, num_holds,
            # num_climbs, dimensions, angle, created_at, updated_at, owner_id,
            # visibility, share_token)
            name = wall[1]
            photo_path = wall[2]
            num_holds = wall[3] or 0
            dimensions = wall[5]
            owner_id = wall[9]
            visibility = wall[10] or "public"
            share_token = wall[11]
            created_at = wall[7]
            updated_at = wall[8]

        # Skip already-migrated (only checkable when the layouts table exists)
        if table_exists(conn, "layouts"):
            exists = conn.execute(
                "SELECT 1 FROM layouts WHERE id=?", (wall_id,)
            ).fetchone()
            if exists:
                skipped += 1
                continue

        width_ft, height_ft = parse_dimensions(dimensions)
        size_name = (
            f"{int(width_ft)}×{int(height_ft)}"
            if (width_ft and height_ft)
            else "Default"
        )
        size_id = f"size-{uuid.uuid4().hex[:12]}"

        print(f"  Migrating wall {wall_id!r} -> layout + size ({size_name})")

        if not dry_run:
            conn.execute(
                """
                INSERT INTO layouts
                    (id, name, description, num_holds, owner_id,
                     visibility, share_token, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (wall_id, name, None, num_holds, owner_id,
                 visibility, share_token, created_at, updated_at),
            )
            conn.execute(
                """
                INSERT INTO sizes
                    (id, layout_id, name, width_ft, height_ft,
                     edge_left, edge_right, edge_bottom, edge_top,
                     photo_path, num_climbs, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (size_id, wall_id, size_name, width_ft, height_ft,
                 0.0, width_ft, 0.0, height_ft,
                 photo_path, 0, created_at, updated_at),
            )
        migrated += 1

    return migrated, skipped


def add_new_columns(conn, dry_run=False):
    """Add layout_id / size_id columns to holds and climbs if missing."""
    cols = [
        ("holds",  "layout_id", "TEXT"),
        ("holds",  "wall_id",   "TEXT"),
        ("climbs", "layout_id", "TEXT"),
        ("climbs", "size_id",   "TEXT"),
        ("climbs", "wall_id",   "TEXT"),
    ]
    for table, col, defn in cols:
        if table_exists(conn, table):
            if not dry_run:
                add_column_if_missing(conn, table, col, defn)
            else:
                if not column_exists(conn, table, col):
                    print(f"  [dry-run] Would add column {table}.{col}")


def backfill_layout_id(conn, dry_run=False):
    """Copy wall_id -> layout_id where layout_id is NULL."""
    for table in ("holds", "climbs"):
        if not table_exists(conn, table):
            continue
        if not (column_exists(conn, table, "wall_id") and
                column_exists(conn, table, "layout_id")):
            continue
        count = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE layout_id IS NULL AND wall_id IS NOT NULL"
        ).fetchone()[0]
        if count:
            print(f"  Backfilling {count} {table} rows: wall_id -> layout_id")
            if not dry_run:
                conn.execute(
                    f"UPDATE {table} SET layout_id = wall_id "
                    f"WHERE layout_id IS NULL AND wall_id IS NOT NULL"
                )


# ---------------------------------------------------------------------------
# Ensure new tables exist
# ---------------------------------------------------------------------------

def ensure_new_tables(conn, dry_run=False):
    """Create layouts and sizes tables if they don't exist yet."""
    if not table_exists(conn, "layouts"):
        print("  Creating `layouts` table")
        if not dry_run:
            conn.execute("""
                CREATE TABLE layouts (
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

    if not table_exists(conn, "sizes"):
        print("  Creating `sizes` table")
        if not dry_run:
            conn.execute("""
                CREATE TABLE sizes (
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


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(conn):
    """Print a summary of migration results."""
    walls = conn.execute("SELECT COUNT(*) FROM walls").fetchone()[0] if table_exists(conn, "walls") else "N/A"
    layouts = conn.execute("SELECT COUNT(*) FROM layouts").fetchone()[0] if table_exists(conn, "layouts") else 0
    sizes = conn.execute("SELECT COUNT(*) FROM sizes").fetchone()[0] if table_exists(conn, "sizes") else 0

    holds_with_lid = 0
    holds_without_lid = 0
    if table_exists(conn, "holds") and column_exists(conn, "holds", "layout_id"):
        holds_with_lid = conn.execute(
            "SELECT COUNT(*) FROM holds WHERE layout_id IS NOT NULL"
        ).fetchone()[0]
        holds_without_lid = conn.execute(
            "SELECT COUNT(*) FROM holds WHERE layout_id IS NULL"
        ).fetchone()[0]

    climbs_with_lid = 0
    climbs_without_lid = 0
    if table_exists(conn, "climbs") and column_exists(conn, "climbs", "layout_id"):
        climbs_with_lid = conn.execute(
            "SELECT COUNT(*) FROM climbs WHERE layout_id IS NOT NULL"
        ).fetchone()[0]
        climbs_without_lid = conn.execute(
            "SELECT COUNT(*) FROM climbs WHERE layout_id IS NULL"
        ).fetchone()[0]

    print("\n=== Migration Verification ===")
    print(f"  walls table rows   : {walls}")
    print(f"  layouts table rows : {layouts}")
    print(f"  sizes table rows   : {sizes}")
    print(f"  holds w/ layout_id : {holds_with_lid}  (missing: {holds_without_lid})")
    print(f"  climbs w/ layout_id: {climbs_with_lid}  (missing: {climbs_without_lid})")

    ok = (
        layouts == walls if isinstance(walls, int) else True
    ) and holds_without_lid == 0 and climbs_without_lid == 0
    print(f"\n  Result: {'PASS' if ok else 'ISSUES FOUND'}")
    return ok


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_migration(db_path: str, dry_run: bool = False):
    print(f"{'[DRY RUN] ' if dry_run else ''}Migrating database: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        print("\n1. Ensuring new tables exist...")
        ensure_new_tables(conn, dry_run)

        print("\n2. Adding new columns to holds/climbs...")
        add_new_columns(conn, dry_run)

        print("\n3. Migrating walls -> layouts + sizes...")
        migrated, skipped = migrate_walls_to_layouts(conn, dry_run)
        print(f"   Migrated: {migrated}  Already done: {skipped}")

        print("\n4. Backfilling layout_id from wall_id...")
        backfill_layout_id(conn, dry_run)

        if not dry_run:
            conn.commit()
            print("\n5. Committed.")
        else:
            conn.rollback()
            print("\n5. Rolled back (dry run).")

        verify(conn)

    except Exception as e:
        conn.rollback()
        print(f"\nERROR: {e}", file=sys.stderr)
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate walls -> layouts + sizes")
    parser.add_argument(
        "--db-path",
        default="data/storage.db",
        help="Path to the SQLite database file (default: data/storage.db)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    args = parser.parse_args()

    if not Path(args.db_path).exists():
        print(f"Database not found: {args.db_path}", file=sys.stderr)
        sys.exit(1)

    run_migration(args.db_path, dry_run=args.dry_run)
