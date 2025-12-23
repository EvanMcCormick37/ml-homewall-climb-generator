"""
API endpoint tests.

These tests define the expected behavior of the API endpoints.
Run with: pytest test_api.py -v

Note: These are specification tests - they define what SHOULD work,
not necessarily what currently works. Use them as a target to build against.
"""
import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

# Adjust this import to match your app structure
from app.main import app

# Path to test photo (relative to this test file)
TEST_PHOTO_PATH = Path(__file__).parent / "test-photo.jpg"


# --- Fixtures ---

@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def sample_holds():
    """Sample hold data for creating a wall."""
    return [
        {"hold_id": 0, "norm_x": 0.2, "norm_y": 0.1, "pull_x": 0.0, "pull_y": 1.0, "useability": 8.0},
        {"hold_id": 1, "norm_x": 0.5, "norm_y": 0.1, "pull_x": 0.0, "pull_y": 1.0, "useability": 7.0},
        {"hold_id": 2, "norm_x": 0.8, "norm_y": 0.1, "pull_x": 0.0, "pull_y": 1.0, "useability": 6.0},
        {"hold_id": 3, "norm_x": 0.3, "norm_y": 0.3, "pull_x": -0.5, "pull_y": 0.5, "useability": 5.0},
        {"hold_id": 4, "norm_x": 0.6, "norm_y": 0.3, "pull_x": 0.5, "pull_y": 0.5, "useability": 5.0},
        {"hold_id": 5, "norm_x": 0.2, "norm_y": 0.5, "pull_x": 0.0, "pull_y": 1.0, "useability": 4.0},
        {"hold_id": 6, "norm_x": 0.5, "norm_y": 0.5, "pull_x": 0.0, "pull_y": 1.0, "useability": 3.0},
        {"hold_id": 7, "norm_x": 0.8, "norm_y": 0.5, "pull_x": 0.0, "pull_y": 1.0, "useability": 4.0},
        {"hold_id": 8, "norm_x": 0.4, "norm_y": 0.8, "pull_x": 0.0, "pull_y": 1.0, "useability": 6.0},
        {"hold_id": 9, "norm_x": 0.6, "norm_y": 0.8, "pull_x": 0.0, "pull_y": 1.0, "useability": 6.0},
    ]


@pytest.fixture
def test_photo():
    """
    Provide test photo file for wall creation.
    Returns a tuple of (filename, file_object, content_type) for use with TestClient.
    """
    if not TEST_PHOTO_PATH.exists():
        # Create a minimal valid JPEG if test photo doesn't exist
        # This is a 1x1 pixel JPEG
        minimal_jpeg = bytes([
            0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
            0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
            0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
            0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
            0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
            0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
            0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
            0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
            0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00,
            0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
            0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03,
            0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
            0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
            0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
            0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72,
            0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
            0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45,
            0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
            0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75,
            0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
            0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3,
            0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
            0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9,
            0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
            0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4,
            0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01,
            0x00, 0x00, 0x3F, 0x00, 0xFB, 0xD5, 0xDB, 0x20, 0xA8, 0xF1, 0x04, 0x12,
            0x56, 0xB6, 0xE1, 0xA4, 0x32, 0x34, 0xDC, 0x7B, 0xDA, 0xFF, 0xD9
        ])
        return ("test-photo.jpg", minimal_jpeg, "image/jpeg")
    
    with open(TEST_PHOTO_PATH, "rb") as f:
        photo_bytes = f.read()
    return ("test-photo.jpg", photo_bytes, "image/jpeg")


def create_wall_with_photo(client, name: str, holds: list, test_photo: tuple, 
                           dimensions: str = None, angle: int = None) -> dict:
    """
    Helper function to create a wall with multipart form data.
    
    Args:
        client: TestClient instance
        name: Wall name
        holds: List of hold dictionaries
        test_photo: Tuple of (filename, file_bytes, content_type)
        dimensions: Optional dimensions string "width,height"
        angle: Optional wall angle
        
    Returns:
        Response JSON as dict
    """
    data = {
        "name": name,
        "holds": json.dumps(holds),
    }
    if dimensions:
        data["dimensions"] = dimensions
    if angle is not None:
        data["angle"] = str(angle)
    
    files = {
        "photo": test_photo
    }
    
    response = client.post("/api/v1/walls", data=data, files=files)
    return response


@pytest.fixture
def sample_wall(client, sample_holds, test_photo):
    """Create a sample wall and return its ID."""
    response = create_wall_with_photo(client, "Test Wall", sample_holds, test_photo)
    assert response.status_code == 201, f"Failed to create wall: {response.text}"
    wall_id = response.json()["id"]
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
            "grade": 4,
            "setter": "test_user",
            "sequence": [[0, 1], [3, 1], [3, 4], [5, 4], [5, 6], [8, 6], [8, 9]],
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
    
    # --- GET /walls ---
    
    def test_list_walls_empty(self, client):
        """List walls when none exist."""
        response = client.get("/api/v1/walls")
        assert response.status_code == 200
        data = response.json()
        assert "walls" in data
        assert "total" in data
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
        assert "num_climbs" in wall
        assert "num_models" in wall
        assert "photo_url" in wall
        assert "created_at" in wall
    
    # --- POST /walls ---
    
    def test_create_wall(self, client, sample_holds, test_photo):
        """Create a new wall with photo."""
        response = create_wall_with_photo(client, "New Wall", sample_holds, test_photo)
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["name"] == "New Wall"
        assert "message" in data
        
        # Cleanup
        client.delete(f"/api/v1/walls/{data['id']}")
    
    def test_create_wall_with_dimensions(self, client, sample_holds, test_photo):
        """Create a new wall with dimensions and angle."""
        response = create_wall_with_photo(
            client, "Wall With Dims", sample_holds, test_photo,
            dimensions="244,305", angle=15
        )
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        
        # Verify dimensions were saved
        wall_resp = client.get(f"/api/v1/walls/{data['id']}")
        wall_data = wall_resp.json()
        assert wall_data["dimensions"] == [244, 305]
        assert wall_data["angle"] == 15
        
        # Cleanup
        client.delete(f"/api/v1/walls/{data['id']}")
    
    def test_create_wall_missing_name(self, client, sample_holds, test_photo):
        """Create wall without name should fail."""
        files = {"photo": test_photo}
        data = {"holds": json.dumps(sample_holds)}
        
        response = client.post("/api/v1/walls", data=data, files=files)
        assert response.status_code == 422
    
    def test_create_wall_missing_photo(self, client, sample_holds):
        """Create wall without photo should fail."""
        data = {
            "name": "No Photo Wall",
            "holds": json.dumps(sample_holds),
        }
        
        response = client.post("/api/v1/walls", data=data)
        assert response.status_code == 422
    
    def test_create_wall_missing_holds(self, client, test_photo):
        """Create wall without holds should fail."""
        files = {"photo": test_photo}
        data = {"name": "No Holds Wall"}
        
        response = client.post("/api/v1/walls", data=data, files=files)
        assert response.status_code == 422
    
    def test_create_wall_invalid_holds_json(self, client, test_photo):
        """Create wall with invalid holds JSON should fail."""
        files = {"photo": test_photo}
        data = {
            "name": "Bad Holds Wall",
            "holds": "not valid json",
        }
        
        response = client.post("/api/v1/walls", data=data, files=files)
        assert response.status_code == 400
    
    def test_create_wall_invalid_photo_type(self, client, sample_holds):
        """Create wall with non-image file should fail."""
        files = {
            "photo": ("test.txt", b"not an image", "text/plain")
        }
        data = {
            "name": "Bad Photo Wall",
            "holds": json.dumps(sample_holds),
        }
        
        response = client.post("/api/v1/walls", data=data, files=files)
        assert response.status_code == 400
    
    # --- GET /walls/{wall_id} ---
    
    def test_get_wall(self, client, sample_wall):
        """Get wall details."""
        response = client.get(f"/api/v1/walls/{sample_wall}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == sample_wall
        assert "name" in data
        assert "holds" in data
        assert isinstance(data["holds"], list)
        assert "num_climbs" in data
        assert "num_models" in data
        assert "dimensions" in data
        assert "angle" in data
        assert "photo_url" in data
        assert "created_at" in data
        assert "updated_at" in data
    
    def test_get_wall_not_found(self, client):
        """Get non-existent wall."""
        response = client.get("/api/v1/walls/wall-nonexistent")
        assert response.status_code == 404
    
    # --- DELETE /walls/{wall_id} ---
    
    def test_delete_wall(self, client, sample_holds, test_photo):
        """Delete a wall."""
        create_resp = create_wall_with_photo(client, "To Delete", sample_holds, test_photo)
        wall_id = create_resp.json()["id"]
        
        response = client.delete(f"/api/v1/walls/{wall_id}")
        assert response.status_code == 204
        
        get_resp = client.get(f"/api/v1/walls/{wall_id}")
        assert get_resp.status_code == 404
    
    def test_delete_wall_not_found(self, client):
        """Delete non-existent wall."""
        response = client.delete("/api/v1/walls/wall-nonexistent")
        assert response.status_code == 404
    
    # --- GET /walls/{wall_id}/photo ---
    
    def test_get_wall_photo(self, client, sample_wall):
        """Get wall photo."""
        response = client.get(f"/api/v1/walls/{sample_wall}/photo")
        assert response.status_code == 200
        assert response.headers["content-type"] in ["image/jpeg", "image/png"]
    
    def test_get_wall_photo_not_found(self, client):
        """Get photo for non-existent wall."""
        response = client.get("/api/v1/walls/wall-nonexistent/photo")
        assert response.status_code == 404


# =============================================================================
# CLIMB ENDPOINTS
# =============================================================================

class TestClimbEndpoints:
    """Tests for /api/v1/walls/{wall_id}/climbs endpoints."""
    
    # --- GET /walls/{wall_id}/climbs ---
    
    def test_list_climbs_empty(self, client, sample_wall):
        """List climbs when none exist."""
        response = client.get(f"/api/v1/walls/{sample_wall}/climbs")
        assert response.status_code == 200
        data = response.json()
        assert "climbs" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data
        assert data["total"] == 0
    
    def test_list_climbs_with_data(self, client, sample_wall, sample_climb):
        """List climbs when climbs exist."""
        response = client.get(f"/api/v1/walls/{sample_wall}/climbs")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        
        climb = data["climbs"][0]
        assert "id" in climb
        assert "wall_id" in climb
        assert "name" in climb
        assert "grade" in climb
        assert "setter" in climb
        assert "sequence" in climb
        assert "tags" in climb
        assert "num_moves" in climb
        assert "created_at" in climb
    
    def test_list_climbs_filter_by_grade(self, client, sample_wall, sample_climb):
        """Filter climbs by grade range."""
        response = client.get(
            f"/api/v1/walls/{sample_wall}/climbs",
            params={"grade_range": "3,5"},
        )
        assert response.status_code == 200
        data = response.json()
        for climb in data["climbs"]:
            if climb["grade"] is not None:
                assert 3 <= climb["grade"] <= 5
    
    def test_list_climbs_filter_by_setter(self, client, sample_wall, sample_climb):
        """Filter climbs by setter."""
        response = client.get(
            f"/api/v1/walls/{sample_wall}/climbs",
            params={"setter": "test_user"},
        )
        assert response.status_code == 200
        data = response.json()
        for climb in data["climbs"]:
            assert climb["setter"] == "test_user"
    
    def test_list_climbs_filter_by_name(self, client, sample_wall, sample_climb):
        """Filter climbs by name substring."""
        response = client.get(
            f"/api/v1/walls/{sample_wall}/climbs",
            params={"name_includes": "Test"},
        )
        assert response.status_code == 200
        data = response.json()
        for climb in data["climbs"]:
            assert "Test" in climb["name"]
    
    def test_list_climbs_filter_by_holds(self, client, sample_wall, sample_climb):
        """Filter climbs that include specific holds."""
        response = client.get(
            f"/api/v1/walls/{sample_wall}/climbs",
            params={"holds_include": [0, 1]},
        )
        assert response.status_code == 200
        data = response.json()
        for climb in data["climbs"]:
            all_holds = [h for pos in climb["sequence"] for h in pos]
            assert 0 in all_holds
            assert 1 in all_holds
    
    def test_list_climbs_filter_by_tags(self, client, sample_wall, sample_climb):
        """Filter climbs that have specific tags."""
        response = client.get(
            f"/api/v1/walls/{sample_wall}/climbs",
            params={"tags_include": ["technical"]},
        )
        assert response.status_code == 200
        data = response.json()
        for climb in data["climbs"]:
            assert "technical" in climb["tags"]
    
    def test_list_climbs_exclude_projects(self, client, sample_wall):
        """Exclude ungraded (project) climbs."""
        client.post(
            f"/api/v1/walls/{sample_wall}/climbs",
            json={"name": "Project", "grade": None, "sequence": [[0, 1], [3, 4]]},
        )
        
        response = client.get(
            f"/api/v1/walls/{sample_wall}/climbs",
            params={"include_projects": False},
        )
        assert response.status_code == 200
        data = response.json()
        for climb in data["climbs"]:
            assert climb["grade"] is not None
    
    def test_list_climbs_pagination(self, client, sample_wall):
        """Test pagination of climbs."""
        for i in range(5):
            client.post(
                f"/api/v1/walls/{sample_wall}/climbs",
                json={"name": f"Climb {i}", "grade": i, "sequence": [[0, 1], [3, 4]]},
            )
        
        response = client.get(
            f"/api/v1/walls/{sample_wall}/climbs",
            params={"limit": 2, "offset": 0},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["climbs"]) == 2
        assert data["limit"] == 2
        assert data["offset"] == 0
    
    # --- POST /walls/{wall_id}/climbs ---
    
    def test_create_climb(self, client, sample_wall):
        """Create a new climb."""
        response = client.post(
            f"/api/v1/walls/{sample_wall}/climbs",
            json={
                "name": "New Climb",
                "grade": 5,
                "setter": "alice",
                "sequence": [[0, 1], [3, 1], [3, 4], [5, 6]],
                "tags": ["powerful"],
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert "message" in data
    
    def test_create_climb_minimal(self, client, sample_wall):
        """Create a climb with only required fields."""
        response = client.post(
            f"/api/v1/walls/{sample_wall}/climbs",
            json={"sequence": [[0, 1], [3, 4]]},
        )
        assert response.status_code == 201
    
    def test_create_climb_invalid_hold(self, client, sample_wall):
        """Create a climb referencing non-existent hold."""
        response = client.post(
            f"/api/v1/walls/{sample_wall}/climbs",
            json={"name": "Invalid", "sequence": [[0, 999]]},
        )
        assert response.status_code in [400, 501]  # Either is acceptable
    
    def test_create_climb_wall_not_found(self, client):
        """Create a climb on non-existent wall."""
        response = client.post(
            "/api/v1/walls/wall-nonexistent/climbs",
            json={"sequence": [[0, 1]]},
        )
        assert response.status_code == 404
    
    # --- DELETE /walls/{wall_id}/climbs/{climb_id} ---
    
    def test_delete_climb(self, client, sample_wall, sample_climb):
        """Delete a climb."""
        response = client.delete(
            f"/api/v1/walls/{sample_wall}/climbs/{sample_climb}"
        )
        assert response.status_code == 200
        assert response.json()["id"] == sample_climb
    
    def test_delete_climb_not_found(self, client, sample_wall):
        """Delete non-existent climb."""
        response = client.delete(
            f"/api/v1/walls/{sample_wall}/climbs/climb-nonexistent"
        )
        assert response.status_code == 404


# =============================================================================
# MODEL ENDPOINTS
# =============================================================================

class TestModelEndpoints:
    """Tests for /api/v1/walls/{wall_id}/models endpoints."""
    
    # --- GET /walls/{wall_id}/models ---
    
    def test_list_models_empty(self, client, sample_wall):
        """List models when none exist."""
        response = client.get(f"/api/v1/walls/{sample_wall}/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "total" in data
        assert data["total"] == 0
    
    # --- POST /walls/{wall_id}/models ---
    
    def test_create_model(self, client, sample_wall, sample_climb):
        """Create and start training a model."""
        response = client.post(
            f"/api/v1/walls/{sample_wall}/models",
            json={
                "model_type": "mlp",
                "features": {
                    "position": True,
                    "pull_direction": True,
                    "difficulty": True,
                },
                "epochs": 10,
                "augment_dataset": True,
            },
        )
        assert response.status_code in [201, 202]  # 202 Accepted is also valid
        data = response.json()
        assert "model_id" in data
        assert "job_id" in data
        assert "message" in data
    
    def test_create_model_minimal(self, client, sample_wall, sample_climb):
        """Create a model with default parameters."""
        response = client.post(
            f"/api/v1/walls/{sample_wall}/models",
            json={},
        )
        assert response.status_code in [201, 202]
    
    def test_create_model_invalid_type(self, client, sample_wall, sample_climb):
        """Create a model with invalid type."""
        response = client.post(
            f"/api/v1/walls/{sample_wall}/models",
            json={"model_type": "invalid_type"},
        )
        assert response.status_code == 422
    
    # --- GET /walls/{wall_id}/models/{model_id} ---
    
    def test_get_model(self, client, sample_wall, sample_climb):
        """Get model details."""
        create_resp = client.post(
            f"/api/v1/walls/{sample_wall}/models",
            json={"model_type": "mlp", "epochs": 5},
        )
        model_id = create_resp.json()["model_id"]
        
        response = client.get(f"/api/v1/walls/{sample_wall}/models/{model_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == model_id
        assert "model_type" in data
        assert "features" in data
        assert "status" in data
        assert "val_loss" in data
        assert "epochs_trained" in data
        assert "created_at" in data
    
    def test_get_model_not_found(self, client, sample_wall):
        """Get non-existent model."""
        response = client.get(
            f"/api/v1/walls/{sample_wall}/models/model-nonexistent"
        )
        assert response.status_code == 404
    
    # --- DELETE /walls/{wall_id}/models/{model_id} ---
    
    def test_delete_model(self, client, sample_wall, sample_climb):
        """Delete a model."""
        create_resp = client.post(
            f"/api/v1/walls/{sample_wall}/models",
            json={"model_type": "mlp", "epochs": 5},
        )
        model_id = create_resp.json()["model_id"]
        
        response = client.delete(
            f"/api/v1/walls/{sample_wall}/models/{model_id}"
        )
        assert response.status_code == 200
        assert response.json()["id"] == model_id
    
    # --- POST /walls/{wall_id}/models/{model_id}/generate ---
    
    def test_generate_climbs(self, client, sample_wall, sample_climb):
        """Generate climbs using a trained model."""
        create_resp = client.post(
            f"/api/v1/walls/{sample_wall}/models",
            json={"model_type": "mlp", "epochs": 5},
        )
        model_id = create_resp.json()["model_id"]
        job_id = create_resp.json()["job_id"]
        
        # Wait for training to complete
        import time
        for _ in range(30):
            job_resp = client.get(f"/api/v1/jobs/{job_id}")
            if job_resp.json()["status"] in ["COMPLETED", "FAILED"]:
                break
            time.sleep(0.5)
        
        response = client.post(
            f"/api/v1/walls/{sample_wall}/models/{model_id}/generate",
            json={
                "starting_holds": [0, 1],
                "max_moves": 5,
                "num_climbs": 3,
                "temperature": 1.0,
                "force_alternating": True,
            },
        )
        
        if job_resp.json()["status"] == "COMPLETED":
            assert response.status_code == 200
            data = response.json()
            assert "model_id" in data
            assert "climbs" in data
            assert "num_generated" in data
            assert len(data["climbs"]) == 3
            
            for climb in data["climbs"]:
                assert "sequence" in climb
                assert "num_moves" in climb
    
    def test_generate_climbs_untrained_model(self, client, sample_wall, sample_climb):
        """Generate climbs with untrained model should fail."""
        create_resp = client.post(
            f"/api/v1/walls/{sample_wall}/models",
            json={"model_type": "mlp", "epochs": 100},
        )
        model_id = create_resp.json()["model_id"]
        
        response = client.post(
            f"/api/v1/walls/{sample_wall}/models/{model_id}/generate",
            json={"starting_holds": [0, 1], "max_moves": 5, "num_climbs": 1},
        )
        assert response.status_code == 400


# =============================================================================
# JOB ENDPOINTS
# =============================================================================

class TestJobEndpoints:
    """Tests for /api/v1/jobs endpoints."""
    
    def test_get_job(self, client, sample_wall, sample_climb):
        """Get job status."""
        create_resp = client.post(
            f"/api/v1/walls/{sample_wall}/models",
            json={"model_type": "mlp", "epochs": 5},
        )
        job_id = create_resp.json()["job_id"]
        
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == job_id
        assert "job_type" in data
        assert "status" in data
        assert "progress" in data
        assert 0.0 <= data["progress"] <= 1.0
        assert data["status"] in ["PENDING", "PROCESSING", "COMPLETED", "FAILED"]
        assert "created_at" in data
    
    def test_get_job_not_found(self, client):
        """Get non-existent job."""
        response = client.get("/api/v1/jobs/job-nonexistent")
        assert response.status_code == 404
    
    def test_job_completion(self, client, sample_wall, sample_climb):
        """Verify job completes successfully."""
        create_resp = client.post(
            f"/api/v1/walls/{sample_wall}/models",
            json={"model_type": "mlp", "epochs": 2},
        )
        job_id = create_resp.json()["job_id"]
        
        import time
        for _ in range(30):
            response = client.get(f"/api/v1/jobs/{job_id}")
            data = response.json()
            
            if data["status"] == "COMPLETED":
                assert data["progress"] == 1.0
                assert data["result"] is not None
                break
            elif data["status"] == "FAILED":
                pytest.fail(f"Job failed: {data.get('error')}")
            
            time.sleep(0.5)
        else:
            pytest.fail("Job did not complete in time")


# =============================================================================
# ERROR HANDLING
# =============================================================================

class TestErrorHandling:
    """Test error responses."""
    
    def test_invalid_json(self, client, sample_wall):
        """Send invalid JSON."""
        response = client.post(
            f"/api/v1/walls/{sample_wall}/climbs",
            content="not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422
    
    def test_missing_required_field(self, client, sample_wall):
        """Missing required field in request."""
        response = client.post(
            f"/api/v1/walls/{sample_wall}/climbs",
            json={"name": "Missing sequence"},
        )
        assert response.status_code == 422
    
    def test_invalid_field_type(self, client, sample_wall):
        """Wrong type for a field."""
        response = client.post(
            f"/api/v1/walls/{sample_wall}/climbs",
            json={"sequence": "not a list"},
        )
        assert response.status_code == 422
    
    def test_method_not_allowed(self, client):
        """Use wrong HTTP method."""
        response = client.patch("/api/v1/walls")
        assert response.status_code == 405


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    def test_full_workflow(self, client, sample_holds, test_photo):
        """Test complete workflow: wall -> climbs -> model -> generate."""
        # 1. Create a wall with photo
        wall_resp = create_wall_with_photo(
            client, "Integration Test Wall", sample_holds, test_photo
        )
        assert wall_resp.status_code == 201
        wall_id = wall_resp.json()["id"]
        
        try:
            # 2. Create some climbs
            climbs = [
                {"sequence": [[0, 1], [3, 1], [3, 4], [5, 4], [5, 6]], "grade": 3},
                {"sequence": [[0, 2], [3, 2], [3, 4], [6, 4], [6, 7]], "grade": 4},
                {"sequence": [[1, 2], [4, 2], [4, 7], [6, 7], [8, 9]], "grade": 5},
            ]
            for climb in climbs:
                resp = client.post(f"/api/v1/walls/{wall_id}/climbs", json=climb)
                assert resp.status_code == 201
            
            # 3. Verify climbs exist
            list_resp = client.get(f"/api/v1/walls/{wall_id}/climbs")
            assert list_resp.json()["total"] == 3
            
            # 4. Create a model
            model_resp = client.post(
                f"/api/v1/walls/{wall_id}/models",
                json={"model_type": "mlp", "epochs": 5},
            )
            assert model_resp.status_code in [201, 202]
            model_id = model_resp.json()["model_id"]
            job_id = model_resp.json()["job_id"]
            
            # 5. Wait for training
            import time
            for _ in range(30):
                job_resp = client.get(f"/api/v1/jobs/{job_id}")
                if job_resp.json()["status"] in ["COMPLETED", "FAILED"]:
                    break
                time.sleep(0.5)
            
            assert job_resp.json()["status"] == "COMPLETED"
            
            # 6. Generate climbs
            gen_resp = client.post(
                f"/api/v1/walls/{wall_id}/models/{model_id}/generate",
                json={"starting_holds": [0, 1], "max_moves": 4, "num_climbs": 2},
            )
            assert gen_resp.status_code == 200
            assert len(gen_resp.json()["climbs"]) == 2
            
            # 7. Verify model appears in list
            models_resp = client.get(f"/api/v1/walls/{wall_id}/models")
            assert models_resp.json()["total"] == 1
            
        finally:
            # Cleanup
            client.delete(f"/api/v1/walls/{wall_id}")
    
    def test_cascade_delete(self, client, sample_holds, test_photo):
        """Deleting a wall should delete associated climbs and models."""
        # Create wall with photo
        wall_resp = create_wall_with_photo(
            client, "Cascade Test", sample_holds, test_photo
        )
        wall_id = wall_resp.json()["id"]
        
        # Create climb
        climb_resp = client.post(
            f"/api/v1/walls/{wall_id}/climbs",
            json={"sequence": [[0, 1], [3, 4]]},
        )
        climb_id = climb_resp.json()["id"]
        
        # Delete wall
        client.delete(f"/api/v1/walls/{wall_id}")
        
        # Verify wall is gone
        assert client.get(f"/api/v1/walls/{wall_id}").status_code == 404
        
        # Verify climbs are gone (if you have a direct climb lookup endpoint)
        # This depends on your API design