"""
Database setup and connection management.
Uses SQLite for simplicity - stores climbs and job queue.
"""
import sqlite3
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

# Database paths
DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "climbs.db"
WALLS_DIR = DATA_DIR / "walls"


def get_db_connection() -> sqlite3.Connection:
    """Get a database connection with row factory."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
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


def init_db():
    """Initialize database tables."""
    DATA_DIR.mkdir(exist_ok=True)
    WALLS_DIR.mkdir(exist_ok=True)
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Walls table - basic metadata (detailed data in JSON files)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS walls (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                photo TEXT NOT NULL,
                dimensions TEXT,
                angle INTEGER,
                num_holds INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Climbs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS climbs (
                id TEXT PRIMARY KEY,
                wall_id TEXT NOT NULL,
                name TEXT,
                grade INTEGER,
                setter TEXT,
                sequence TEXT NOT NULL,  -- serialized list of positions
                tags TEXT,               -- serialized list of tags
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (wall_id) REFERENCES walls(id) ON DELETE CASCADE
            )
        """)
        
        # Models table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                wall_id TEXT NOT NULL,
                model_type TEXT NOT NULL,
                features TEXT NOT NULL,  -- JSON object of feature flags
                status TEXT DEFAULT 'untrained',
                moves_trained INTEGER DEFAULT 0,
                climbs_trained INTEGER DEFAULT 0,
                val_loss REAL,
                epochs_trained INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                trained_at TIMESTAMP,
                FOREIGN KEY (wall_id) REFERENCES walls(id) ON DELETE CASCADE
            )
        """)
        
        # Jobs table - simple queue for background tasks
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                job_type TEXT NOT NULL,
                status TEXT DEFAULT 'PENDING',
                progress REAL DEFAULT 0.0,
                params TEXT,             -- JSON object of job parameters
                result TEXT,             -- JSON object of job result
                error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP
            )
        """)
        
        # Indexes for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_climbs_wall ON climbs(wall_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_climbs_setter ON climbs(setter)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_wall ON models(wall_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
