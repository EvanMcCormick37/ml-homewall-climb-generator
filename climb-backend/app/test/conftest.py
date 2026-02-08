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
    
    Instead of deleting the entire DATA_DIR (which kills the models),
    we only delete the database file and the walls directory.
    """
    data_dir = Path(settings.DATA_DIR)
    walls_dir = settings.WALLS_DIR
    db_path = settings.DB_PATH

    # but specifically walls for the app)
    walls_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Initialize database
    init_db()
    
    yield


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)

@pytest.fixture(autouse=True)
def clean_db_between_tests():
    """Reset database rows between tests without deleting files."""
    yield