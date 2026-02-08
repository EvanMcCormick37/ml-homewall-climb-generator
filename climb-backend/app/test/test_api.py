"""
API endpoint tests.

These tests define the expected behavior of the API endpoints.
Run with: pytest test_api.py -v
"""
import time
import json
import pytest
from fastapi.testclient import TestClient

# Adjust this import to match your app structure
from app.main import app
from app.config import settings


# --- Fixtures ---

@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def sample_holds():
    """Sample hold data for creating a wall."""
    return [
        {"hold_index": 0, "x": 0.2, "y": 0.1, "pull_x": 0.0, "pull_y": 1.0, "useability": 0.8},
        {"hold_index": 1, "x": 0.5, "y": 0.1, "pull_x": 0.0, "pull_y": 1.0, "useability": 0.7},
        {"hold_index": 2, "x": 0.8, "y": 0.1, "pull_x": 0.0, "pull_y": 1.0, "useability": 0.6},
        {"hold_index": 3, "x": 0.3, "y": 0.3, "pull_x": -0.5, "pull_y": 0.5, "useability": 0.5},
        {"hold_index": 4, "x": 0.6, "y": 0.3, "pull_x": 0.5, "pull_y": 0.5, "useability": 0.5},
        {"hold_index": 5, "x": 0.2, "y": 0.5, "pull_x": 0.0, "pull_y": 1.0, "useability": 0.4},
        {"hold_index": 6, "x": 0.5, "y": 0.5, "pull_x": 0.0, "pull_y": 1.0, "useability": 0.3},
        {"hold_index": 7, "x": 0.8, "y": 0.5, "pull_x": 0.0, "pull_y": 1.0, "useability": 0.4},
        {"hold_index": 8, "x": 0.4, "y": 0.8, "pull_x": 0.0, "pull_y": 1.0, "useability": 0.6},
        {"hold_index": 9, "x": 0.6, "y": 0.8, "pull_x": 0.0, "pull_y": 1.0, "useability": 0.6},
    ]


@pytest.fixture
def test_photo():
    """
    Provide test photo file for wall creation.
    Returns a tuple of (filename, file_object, content_type) for use with TestClient.
    """
    # Ensure this file exists in your test assets or mock it if strictly needed
    # For now assuming the file exists as per previous context
    try:
        with open(settings.TEST_ASSETS_DIR / "test-photo.jpg", "rb") as f:
            photo_bytes = f.read()
        return ("test-photo.jpg", photo_bytes, "image/jpeg")
    except FileNotFoundError:
        # Fallback for environments without the asset
        return ("test-photo.jpg", b"fake image bytes", "image/jpeg")


def create_wall_with_photo(client, name: str, test_photo: tuple, 
                           dimensions: str = None, angle: int = None) -> dict:
    """Helper function to create a wall with multipart form data."""
    data = {
        "name": name,
    }
    data["dimensions"] = dimensions
    if angle is not None:
        data["angle"] = str(angle)
    else:
        data["angle"] = None
    files = {
        "photo": test_photo
    }
    
    response = client.post("/api/v1/walls", data=data, files=files)
    return response


def set_wall_holds(client, wall_id: str, holds: list) -> dict:
    """Helper function to set holds on a wall."""
    response = client.put(
        f"/api/v1/walls/{wall_id}/holds",
        data={"holds": json.dumps(holds)}
    )
    return response


@pytest.fixture
def sample_wall(client, sample_holds, test_photo):
    """Create a sample wall with holds and return its ID."""
    # Create wall
    response = create_wall_with_photo(client, "Test Wall", test_photo)
    assert response.status_code == 201, f"Failed to create wall: {response.text}"
    wall_id = response.json()["id"]
    
    # Set holds on wall
    holds_response = set_wall_holds(client, wall_id, sample_holds)
    assert holds_response.status_code == 201, f"Failed to set holds: {holds_response.text}"
    
    yield wall_id
    # Cleanup
    client.delete(f"/api/v1/walls/{wall_id}")


@pytest.fixture
def sample_climb(client, sample_wall):
    """Create a sample climb and return its ID."""
    response = client.post(
        f"/api/v1/walls/{sample_wall}/climbs",
        json={
            "name": "Test Climb",
            "grade": 40,
            "setter_name": "test_user",
            "angle": 40,
            "holdset": {
                "start": [0, 1],
                "finish": [8, 9],
                "hand": [3, 4, 5, 6],
                "foot": [2, 7],
            },
            "tags": ["technical", "balance"],
        },
    )
    assert response.status_code == 201
    return response.json()["id"]


# =============================================================================
# WALL ENDPOINTS
# =============================================================================

class TestWallEndpoints:
    """Tests for /api/v1/walls endpoints."""
    
    def test_list_walls_empty(self, client):
        """List walls when none exist."""
        response = client.get("/api/v1/walls")
        assert response.status_code == 200
        data = response.json()
        assert "walls" in data
        assert isinstance(data["walls"], list)
    
    def test_list_walls_with_data(self, client, sample_wall):
        """List walls when walls exist."""
        response = client.get("/api/v1/walls")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        
        wall = next((w for w in data["walls"] if w["id"] == sample_wall), None)
        assert wall is not None
        assert "name" in wall
        assert "num_holds" in wall
    
    def test_create_wall(self, client, test_photo):
        """Create a new wall with photo."""
        response = create_wall_with_photo(client, "New Wall", test_photo)
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["name"] == "New Wall"
        client.delete(f"/api/v1/walls/{data['id']}")
    
    def test_create_wall_with_dimensions(self, client, test_photo):
        """Create a new wall with dimensions and angle."""
        response = create_wall_with_photo(
            client, "Wall With Dims", test_photo,
            dimensions="244,305", angle=15
        )
        assert response.status_code == 201
        data = response.json()
        
        # Verify dimensions were saved
        wall_resp = client.get(f"/api/v1/walls/{data['id']}")
        wall_data = wall_resp.json()["metadata"]
        assert wall_data["dimensions"] == [244, 305]
        assert wall_data["angle"] == 15
        client.delete(f"/api/v1/walls/{data['id']}")
    
    def test_set_holds(self, client, test_photo, sample_holds):
        """Set holds on a wall."""
        response = create_wall_with_photo(client, "Holds Test Wall", test_photo)
        wall_id = response.json()["id"]
        
        holds_response = set_wall_holds(client, wall_id, sample_holds)
        assert holds_response.status_code == 201
        
        wall_resp = client.get(f"/api/v1/walls/{wall_id}")
        assert len(wall_resp.json()["holds"]) == len(sample_holds)
        client.delete(f"/api/v1/walls/{wall_id}")

    def test_get_wall(self, client, sample_wall):
        """Get wall details."""
        response = client.get(f"/api/v1/walls/{sample_wall}")
        assert response.status_code == 200
        assert response.json()["metadata"]["id"] == sample_wall
    
    def test_delete_wall(self, client, test_photo):
        """Delete a wall."""
        create_resp = create_wall_with_photo(client, "To Delete", test_photo)
        wall_id = create_resp.json()["id"]
        
        response = client.delete(f"/api/v1/walls/{wall_id}")
        assert response.status_code == 204
        assert client.get(f"/api/v1/walls/{wall_id}").status_code == 404


# =============================================================================
# CLIMB ENDPOINTS
# =============================================================================

class TestClimbEndpoints:
    """Tests for /api/v1/walls/{wall_id}/climbs endpoints."""
    
    def test_list_climbs_empty(self, client, sample_wall):
        """List climbs when none exist."""
        response = client.get(f"/api/v1/walls/{sample_wall}/climbs")
        assert response.status_code == 200
        assert response.json()["total"] == 0
    
    def test_create_climb(self, client, sample_wall):
        """Create a new climb."""
        response = client.post(
            f"/api/v1/walls/{sample_wall}/climbs",
            json={
                "name": "New Climb",
                "grade": 50,
                "setter_name": "alice",
                "angle": 45,
                "holdset": {
                    "start": [0, 1],
                    "finish": [5, 6],
                    "hand": [3, 4],
                    "foot": [2],
                },
                "tags": ["powerful"],
            },
        )
        assert response.status_code == 201
        assert "id" in response.json()
    
    def test_list_climbs_filters(self, client, sample_wall, sample_climb):
        """Test various list filters."""
        # By Grade
        resp = client.get(f"/api/v1/walls/{sample_wall}/climbs", params={"grade_range": [30, 50]})
        assert resp.status_code == 200
        assert len(resp.json()["climbs"]) > 0

        # By Name
        resp = client.get(f"/api/v1/walls/{sample_wall}/climbs", params={"name_includes": "Test"})
        assert resp.status_code == 200
        assert len(resp.json()["climbs"]) > 0

    def test_delete_climb(self, client, sample_wall, sample_climb):
        """Delete a climb."""
        response = client.delete(f"/api/v1/walls/{sample_wall}/climbs/{sample_climb}")
        assert response.status_code == 200
        assert response.json()["id"] == sample_climb


# =============================================================================
# GENERATE ENDPOINTS (NEW)
# =============================================================================

class TestGenerateEndpoints:
    """Tests for /api/v1/walls/{wall_id}/generate endpoints."""

    def test_generate_climbs_success(self, client, sample_wall):
        """Test successful climb generation."""
        payload = {
            "num_climbs": 1,
            "grade": "V4",
            "grade_scale": "v_grade",
            "angle": 40,
            "deterministic": False
        }
        
        # Note: The router path prefix ends in .../generate, so we post to that base
        response = client.post(
            f"/api/v1/walls/{sample_wall}/generate",
            json=payload
        )
        
        assert response.status_code == 200, f"Generation failed: {response.text}"
        data = response.json()
        print(data)
        
        # Verify Top Level Response
        assert data["wall_id"] == sample_wall
        assert data["num_generated"] == 1
        assert isinstance(data["climbs"], list)
        assert len(data["climbs"]) == 1
        
        # Verify Echoed Parameters
        assert data["parameters"]["grade"] == "V4"
        assert data["parameters"]["angle"] == 40