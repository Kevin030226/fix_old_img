"""Models API router (plan §3.8, §3.15).

    GET  /api/v1/models                    list declared + runnable models
    GET  /api/v1/models/health             per-model health probe
    GET  /api/v1/models/{name}             one model
    GET  /api/v1/models/{name}/versions    resident/active/previous versions
    POST /api/v1/models/{name}/reload      hot swap to a version (admin)
    POST /api/v1/models/{name}/rollback    return to the previous version (admin)
    POST /api/v1/models/{name}/unload      free memory (admin)
    POST /api/v1/models/{name}/warmup      explicit warmup (admin)

Admin-only routes are the write side: they can evict a model from GPU memory or
change which version serves traffic, so they require the ``models:write`` scope.
"""
from fastapi import APIRouter, Depends, Query

from fiximg.api.dependencies import Principal, require_admin, require_principal
from fiximg.api.schemas.model import (
    ModelHealthResponse,
    ModelListResponse,
    ModelSwitchResponse,
    ModelVersionsResponse,
    ModelView,
)
from fiximg.application.model_service import model_service

router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.get("", response_model=ModelListResponse, summary="List models")
async def list_models(_principal: Principal = Depends(require_principal)):
    return ModelListResponse(models=[ModelView(**m) for m in model_service.list_models()])


@router.get("/health", response_model=ModelHealthResponse, summary="Model health")
async def models_health(_principal: Principal = Depends(require_principal)):
    return ModelHealthResponse(**model_service.health())


@router.get("/{name}", response_model=ModelView, summary="Describe one model")
async def get_model(name: str, _principal: Principal = Depends(require_principal)):
    return ModelView(**model_service.get_model(name))


@router.get(
    "/{name}/versions",
    response_model=ModelVersionsResponse,
    summary="Resident versions of a model",
)
async def model_versions(name: str, _principal: Principal = Depends(require_principal)):
    """Which versions are in memory, which serves traffic, which is the fallback."""
    return ModelVersionsResponse(**model_service.versions(name))


@router.post(
    "/{name}/reload",
    response_model=ModelSwitchResponse,
    summary="Hot swap a model to a version",
)
async def reload_model(
    name: str,
    version: str | None = Query(
        default=None, description="Target version; defaults to the manifest declaration"
    ),
    _principal: Principal = Depends(require_admin),
):
    """Zero-downtime version switch (plan §3.15).

    The candidate is built, validated, warmed and health-checked while the
    current version keeps serving; routing switches atomically afterwards. A
    failing candidate returns ``switched: false`` and changes nothing.
    """
    return ModelSwitchResponse(**model_service.reload(name, version))


@router.post(
    "/{name}/rollback",
    response_model=ModelSwitchResponse,
    summary="Roll back to the previous version",
)
async def rollback_model(name: str, _principal: Principal = Depends(require_admin)):
    return ModelSwitchResponse(**model_service.rollback(name))


@router.post("/{name}/unload", response_model=ModelView, summary="Unload a model")
async def unload_model(name: str, _principal: Principal = Depends(require_admin)):
    return ModelView(**model_service.unload(name))


@router.post("/{name}/warmup", response_model=ModelView, summary="Warm up a model")
async def warmup_model(name: str, _principal: Principal = Depends(require_admin)):
    return ModelView(**model_service.warmup(name))
