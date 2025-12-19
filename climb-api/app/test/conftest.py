import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database import init_db, DB_PATH
import os

@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    '''Initialize test database.'''
    # Use a test database
    os.environ["DATA_DIR"] = "test_data"
    init_db()
    yield
    # Cleanup after all tests
    import shutil
    if os.path.exists("test_data"):
        shutil.rmtree("test_data")

@pytest.fixture
def client():
    '''Create test client.'''
    return TestClient(app)