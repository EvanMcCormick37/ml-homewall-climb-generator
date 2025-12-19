"""
Router for ML model-related endpoints.

Endpoints:
- GET  /walls/{wall_id}/models                     - List models
- POST /walls/{wall_id}/models                     - Create and train a model
- GET  /walls/{wall_id}/models/{model_id}          - Get model details
- DELETE /walls/{wall_id}/models/{model_id}        - Delete a model
- POST /walls/{wall_id}/models/{model_id}/generate - Generate climbs
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, status

from app.schemas import (
    ModelCreate,
    ModelDetail,
    ModelListResponse,
    ModelCreateResponse,
    ModelDeleteResponse,
    GenerateRequest,
    GenerateResponse,
)
from app.services.model_service import ModelService
from app.services.job_service import JobService

router = APIRouter()
model_service = ModelService()
job_service = JobService()


@router.get(
    "",
    response_model=ModelListResponse,
    summary="List models",
    description="Get all models for a wall.",
)
async def list_models(wall_id: str):
    """List all models for a wall."""
    # TODO: Implement
    # models = model_service.get_models_for_wall(wall_id)
    # return ModelListResponse(models=models, total=len(models))
    raise HTTPException(status_code=501, detail="Not implemented")


@router.post(
    "",
    response_model=ModelCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create and train a model",
    description="Create a new model and start training in the background.",
)
async def create_model(
    wall_id: str,
    model_config: ModelCreate,
    background_tasks: BackgroundTasks,
):
    """
    Create a new model and start training.
    
    Training runs as a background job. Use the returned job_id to poll
    for status via GET /jobs/{job_id}.
    """
    # TODO: Implement
    # 1. Validate wall exists
    # if not wall_service.wall_exists(wall_id):
    #     raise HTTPException(status_code=404, detail="Wall not found")
    
    # 2. Create model record
    # model_id = model_service.create_model(wall_id, model_config)
    
    # 3. Create job record
    # job_id = job_service.create_job(
    #     job_type="train_model",
    #     params={"model_id": model_id, "wall_id": wall_id, **model_config.dict()}
    # )
    
    # 4. Start background training task
    # background_tasks.add_task(
    #     model_service.train_model_task,
    #     job_id=job_id,
    #     model_id=model_id,
    #     wall_id=wall_id,
    #     config=model_config,
    # )
    
    # return ModelCreateResponse(model_id=model_id, job_id=job_id)
    raise HTTPException(status_code=501, detail="Not implemented")


@router.get(
    "/{model_id}",
    response_model=ModelDetail,
    summary="Get model details",
    description="Get detailed information about a model.",
)
async def get_model(wall_id: str, model_id: str):
    """Get model details including training stats."""
    # TODO: Implement
    # model = model_service.get_model(wall_id, model_id)
    # if not model:
    #     raise HTTPException(status_code=404, detail="Model not found")
    # return model
    raise HTTPException(status_code=501, detail="Not implemented")


@router.delete(
    "/{model_id}",
    response_model=ModelDeleteResponse,
    summary="Delete a model",
    description="Delete a model and its trained weights.",
)
async def delete_model(wall_id: str, model_id: str):
    """Delete a model."""
    # TODO: Implement
    # success = model_service.delete_model(wall_id, model_id)
    # if not success:
    #     raise HTTPException(status_code=404, detail="Model not found")
    # return ModelDeleteResponse(id=model_id)
    raise HTTPException(status_code=501, detail="Not implemented")


@router.post(
    "/{model_id}/generate",
    response_model=GenerateResponse,
    summary="Generate climbs",
    description="Use a trained model to generate climb sequences.",
)
async def generate_climbs(
    wall_id: str,
    model_id: str,
    request: GenerateRequest,
):
    """
    Generate climbs using a trained model.
    
    Parameters:
    - starting_holds: [left_hand_hold_id, right_hand_hold_id]
    - max_moves: Maximum moves per climb
    - num_climbs: Number of climbs to generate
    - temperature: Sampling temperature (higher = more random)
    - force_alternating: Require alternating limb movement
    - features: Which features to consider for hold selection
    """
    # TODO: Implement
    # 1. Validate model exists and is trained
    # model = model_service.get_model(wall_id, model_id)
    # if not model:
    #     raise HTTPException(status_code=404, detail="Model not found")
    # if model.status != "trained":
    #     raise HTTPException(status_code=400, detail="Model is not trained")
    
    # 2. Validate starting holds exist in wall
    # ...
    
    # 3. Load model and generate
    # generated = model_service.generate_climbs(model_id, request)
    
    # return GenerateResponse(
    #     model_id=model_id,
    #     climbs=generated,
    #     num_generated=len(generated),
    #     parameters=request,
    # )
    raise HTTPException(status_code=501, detail="Not implemented")
