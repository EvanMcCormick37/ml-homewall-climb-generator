"""
API endpoint tests.

These tests define the expected behavior of the API endpoints.
Run with: pytest test_api.py -v

Note: These are specification tests - they define what SHOULD work,
not necessarily what currently works. Use them as a target to build against.
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
    with open(settings.TEST_ASSETS_DIR / "test-photo.jpg", "rb") as f:
        photo_bytes = f.read()
    return ("test-photo.jpg", photo_bytes, "image/jpeg")


def create_wall_with_photo(client, name: str, test_photo: tuple, 
                           dimensions: str = None, angle: int = None) -> dict:
    """
    Helper function to create a wall with multipart form data.
    
    Args:
        client: TestClient instance
        name: Wall name
        test_photo: Tuple of (filename, file_bytes, content_type)
        dimensions: Optional dimensions string "width,height"
        angle: Optional wall angle
        
    Returns:
        Response object
    """
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
    """
    Helper function to set holds on a wall.
    
    Args:
        client: TestClient instance
        wall_id: Wall ID
        holds: List of hold dictionaries
        
    Returns:
        Response object
    """
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


@pytest.fixture
def add_sample_climbs(client, sample_wall):
    """Add multiple sample climbs to the sample wall."""
    for i in range(25):
        response = client.post(
            f"/api/v1/walls/{sample_wall}/climbs",
            json={
                "name": f"Test Climb {i}",
                "grade": i * 4,  # Spread grades across range
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
    
    def test_create_wall(self, client, test_photo):
        """Create a new wall with photo."""
        response = create_wall_with_photo(client, "New Wall", test_photo)
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["name"] == "New Wall"
        
        # Cleanup
        client.delete(f"/api/v1/walls/{data['id']}")
    
    def test_create_wall_with_dimensions(self, client, test_photo):
        """Create a new wall with dimensions and angle."""
        response = create_wall_with_photo(
            client, "Wall With Dims", test_photo,
            dimensions="244,305", angle=15
        )
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        
        # Verify dimensions were saved
        wall_resp = client.get(f"/api/v1/walls/{data['id']}")
        wall_detail = wall_resp.json()
        wall_data = wall_detail["metadata"]
        assert wall_data["dimensions"] == [244, 305]
        assert wall_data["angle"] == 15
        
        # Cleanup
        client.delete(f"/api/v1/walls/{data['id']}")
    
    def test_create_wall_without_angle(self,client, test_photo):
        """Create a wall without declaring a default angle"""
        response = create_wall_with_photo(
            client, "Wall With Dims", test_photo,
            dimensions="244,305"
        )
        assert response.status_code == 201
    
    def test_create_wall_missing_name(self, client, test_photo):
        """Create wall without name should fail."""
        files = {"photo": test_photo}
        data = {}
        
        response = client.post("/api/v1/walls", data=data, files=files)
        assert response.status_code == 422
    
    def test_create_wall_missing_photo(self, client):
        """Create wall without photo should fail."""
        data = {
            "name": "No Photo Wall",
        }
        
        response = client.post("/api/v1/walls", data=data)
        assert response.status_code == 422
    
    def test_create_wall_invalid_photo_type(self, client):
        """Create wall with non-image file should fail."""
        files = {
            "photo": ("test.txt", b"not an image", "text/plain")
        }
        data = {
            "name": "Bad Photo Wall",
        }
        
        response = client.post("/api/v1/walls", data=data, files=files)
        assert response.status_code == 400
    
    # --- PUT /walls/{wall_id}/holds ---
    
    def test_set_holds(self, client, test_photo, sample_holds):
        """Set holds on a wall."""
        # Create wall first
        response = create_wall_with_photo(client, "Holds Test Wall", test_photo)
        assert response.status_code == 201
        wall_id = response.json()["id"]
        
        # Set holds
        holds_response = set_wall_holds(client, wall_id, sample_holds)
        assert holds_response.status_code == 201
        
        # Verify holds were saved
        wall_resp = client.get(f"/api/v1/walls/{wall_id}")
        wall_detail = wall_resp.json()
        assert len(wall_detail["holds"]) == len(sample_holds)
        assert wall_detail["metadata"]["num_holds"] == len(sample_holds)
        
        # Cleanup
        client.delete(f"/api/v1/walls/{wall_id}")
    
    def test_set_holds_replace(self, client, sample_wall, sample_holds):
        """Replace existing holds on a wall."""
        # Create new smaller holdset
        new_holds = sample_holds[:5]
        
        holds_response = set_wall_holds(client, sample_wall, new_holds)
        assert holds_response.status_code == 201
        
        # Verify holds were replaced
        wall_resp = client.get(f"/api/v1/walls/{sample_wall}")
        wall_detail = wall_resp.json()
        assert len(wall_detail["holds"]) == 5
        assert wall_detail["metadata"]["num_holds"] == 5
    
    def test_set_holds_invalid_json(self, client, sample_wall):
        """Set holds with invalid JSON should fail."""
        response = client.put(
            f"/api/v1/walls/{sample_wall}/holds",
            data={"holds": "not valid json"}
        )
        assert response.status_code == 400
    
    def test_set_holds_wall_not_found(self, client, sample_holds):
        """Set holds on non-existent wall should fail."""
        response = client.put(
            "/api/v1/walls/wall-nonexistent/holds",
            data={"holds": json.dumps(sample_holds)}
        )
        assert response.status_code == 404
    
    # --- GET /walls/{wall_id} ---
    
    def test_get_wall(self, client, sample_wall):
        """Get wall details."""
        response = client.get(f"/api/v1/walls/{sample_wall}")
        assert response.status_code == 200
        wall_detail = response.json()
        assert isinstance(wall_detail["holds"], list)
        assert "holds" in wall_detail
        assert "metadata" in wall_detail
        metadata = wall_detail["metadata"]
        assert metadata["id"] == sample_wall
        assert "name" in metadata
        assert "num_climbs" in metadata
        assert "num_models" in metadata
        assert "dimensions" in metadata
        assert "angle" in metadata
        assert "photo_url" in metadata
        assert "created_at" in metadata
        assert "updated_at" in metadata
    
    def test_get_wall_not_found(self, client):
        """Get non-existent wall."""
        response = client.get("/api/v1/walls/wall-nonexistent")
        assert response.status_code == 404
    
    # --- DELETE /walls/{wall_id} ---
    
    def test_delete_wall(self, client, sample_holds, test_photo):
        """Delete a wall."""
        create_resp = create_wall_with_photo(client, "To Delete", test_photo)
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
        if response.status_code == 422:
            print(response.json())
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
        if response.status_code == 422:
            print(response.json())
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        
        climb = data["climbs"][0]
        assert "id" in climb
        assert "wall_id" in climb
        assert "name" in climb
        assert "grade" in climb
        assert "setter_name" in climb
        assert "holdset" in climb
        assert "tags" in climb
        assert "angle" in climb
        assert "ascents" in climb
        assert "created_at" in climb
    
    def test_list_climbs_filter_by_grade(self, client, sample_wall, sample_climb):
        """Filter climbs by grade range."""
        response = client.get(
            f"/api/v1/walls/{sample_wall}/climbs",
            params={"grade_range": [30, 50]},
        )
        assert response.status_code == 200
        data = response.json()
        for climb in data["climbs"]:
            if climb["grade"] is not None:
                assert 30 <= climb["grade"] <= 50
    
    def test_list_climbs_filter_by_setter(self, client, sample_wall, sample_climb):
        """Filter climbs by setter."""
        response = client.get(
            f"/api/v1/walls/{sample_wall}/climbs",
            params={"setter_name": "test_user"},
        )
        assert response.status_code == 200
        data = response.json()
        for climb in data["climbs"]:
            assert climb["setter_name"] == "test_user"
    
    def test_list_climbs_filter_by_angle(self, client, sample_wall, sample_climb):
        """Filter climbs by wall angle."""
        response = client.get(
            f"/api/v1/walls/{sample_wall}/climbs",
            params={"angle": 40},
        )
        assert response.status_code == 200
        data = response.json()
        for climb in data["climbs"]:
            assert climb["angle"] == 40
    
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
            holdset = climb["holdset"]
            hold_indices = [
                h_idx
                for role in holdset.values()
                for h_idx in role
            ]
            assert 0 in hold_indices
            assert 1 in hold_indices
    
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
            json={
                "name": "Project",
                "grade": None,
                "angle": 40,
                "holdset": {
                    "start": [0, 1],
                    "finish": [3, 4],
                    "hand": [],
                    "foot": [],
                },
            },
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
                json={
                    "name": f"Climb {i}",
                    "grade": i * 10,
                    "angle": 40,
                    "holdset": {
                        "start": [0, 1],
                        "finish": [3, 4],
                        "hand": [],
                        "foot": [],
                    },
                },
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
        data = response.json()
        assert "id" in data
        
    
    def test_create_climb_minimal(self, client, sample_wall):
        """Create a climb with only required fields."""
        response = client.post(
            f"/api/v1/walls/{sample_wall}/climbs",
            json={
                "name": "climb regcedfdsfa",
                "angle": 40,
                "holdset": {
                    "start": [0, 1],
                    "finish": [3, 4],
                    "hand": [],
                    "foot": [],
                },
            },
        )
        assert response.status_code == 201
    
    def test_create_climb_wall_not_found(self, client):
        """Create a climb on non-existent wall."""
        response = client.post(
            "/api/v1/walls/wall-nonexistent/climbs",
            json={
                "name": "climb regcedfdsfa",
                "angle": 40,
                "holdset": {
                    "start": [0, 1],
                    "finish": [3, 4],
                    "hand": [],
                    "foot": [],
                },
            },
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

@pytest.mark.skip(reason="Models are about to be overhauled. No point in testing them here.")
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
        resp = create_resp.json()
        model_id = resp["model_id"]
        job_id = resp["job_id"]

        # Wait for training to complete before attempting deletion
        for _ in range(30):
            job_resp = client.get(f"/api/v1/jobs/{job_id}")
            if job_resp.json()["status"] in ["COMPLETED", "FAILED"]:
                break
        
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


# =============================================================================
# JOB ENDPOINTS
# =============================================================================
@pytest.mark.skip(reason="Models being rebuilt, so job service is obselete for now.")
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
    
    def test_job_completion(self, client, sample_wall, add_sample_climbs):
        """Verify job completes successfully."""
        create_resp = client.post(
            f"/api/v1/walls/{sample_wall}/models",
            json={"epochs": 2},
        )
        job_id = create_resp.json()["job_id"]
        
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
            json={"name": "Missing holds"},
        )
        assert response.status_code == 422
    
    def test_invalid_field_type(self, client, sample_wall):
        """Wrong type for a field."""
        response = client.post(
            f"/api/v1/walls/{sample_wall}/climbs",
            json={"holdset": "not a dict"},
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
        """Test complete workflow: wall -> holds -> climbs -> model -> generate."""
        # 1. Create a wall with photo
        wall_resp = create_wall_with_photo(
            client, "Integration Test Wall", test_photo
        )
        assert wall_resp.status_code == 201
        wall_id = wall_resp.json()["id"]
        
        try:
            # 2. Set holds on wall
            holds_resp = set_wall_holds(client, wall_id, sample_holds)
            assert holds_resp.status_code == 201
            
            # 3. Create some climbs
            climbs = [
                {  
                    "name":"Gordon was a true bro",
                    "angle": 40,
                    "grade": 30,
                    "holdset": {"start": [0, 1], "finish": [5, 6], "hand": [3, 4], "foot": []},
                },
                {  
                    "name":"Gordon was a true bro",
                    "angle": 40,
                    "grade": 40,
                    "holdset": {"start": [0, 2], "finish": [6, 7], "hand": [3, 4], "foot": []},
                },
                {  
                    "name":"Gordon was a true bro",
                    "angle": 40,
                    "grade": 50,
                    "holdset": {"start": [1, 2], "finish": [8, 9], "hand": [4, 6, 7], "foot": []},
                },
            ]
            for climb in climbs:
                resp = client.post(f"/api/v1/walls/{wall_id}/climbs", json=climb)
                assert resp.status_code == 201
            
            # 4. Verify climbs exist
            list_resp = client.get(f"/api/v1/walls/{wall_id}/climbs")
            assert list_resp.json()["total"] == 3
            
        finally:
            # Cleanup
            client.delete(f"/api/v1/walls/{wall_id}")
    
    def test_cascade_delete(self, client, sample_holds, test_photo):
        """Deleting a wall should delete associated climbs, models, and holds."""
        # Create wall with photo
        wall_resp = create_wall_with_photo(
            client, "Cascade Test", test_photo
        )
        wall_id = wall_resp.json()["id"]
        
        # Set holds
        set_wall_holds(client, wall_id, sample_holds)
        
        # Create climb
        climb_resp = client.post(
            f"/api/v1/walls/{wall_id}/climbs",
            json={
                "name": "Evan knows the secret code to reverse gravity. And he's watching you right now. Tread lightly upon this hallowed code.",
                "angle": 40,
                "holdset": {"start": [0, 1], "finish": [3, 4], "hand": [], "foot": []},
            },
        )
        climb_id = climb_resp.json()["id"]
        
        # Delete wall
        client.delete(f"/api/v1/walls/{wall_id}")
        
        # Verify wall is gone
        assert client.get(f"/api/v1/walls/{wall_id}").status_code == 404