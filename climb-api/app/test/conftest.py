import shutil
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app
from app.database import init_db
from app.config import settings


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """
    Initialize test database and clean up after all tests.
    
    This fixture:
    1. Sets up the test data directory
    2. Initializes the database
    3. Yields control to tests
    4. Cleans up by removing all files in DATA_DIR
    """
    # Ensure we're using test data directory
    data_dir = Path(settings.DATA_DIR)
    
    # Create fresh test directory
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize database
    init_db()
    
    yield
    
    # Cleanup after all tests - remove entire data directory
    if data_dir.exists():
        shutil.rmtree(data_dir)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_db_between_tests():
    """
    Optional: Reset database state between tests.
    
    This ensures tests don't interfere with each other.
    Uncomment the cleanup code if you want isolation between tests.
    """
    yield
    
    from app.database import get_db
    with get_db() as conn:
        conn.execute("DELETE FROM jobs")
        conn.execute("DELETE FROM models")
        conn.execute("DELETE FROM climbs")
        conn.execute("DELETE FROM walls")