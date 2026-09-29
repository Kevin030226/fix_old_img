"""S3 / MinIO artifact store (plan §2.8 / §3.13, P2).

Implemented against the S3 API surface shared by AWS S3, MinIO and most object
stores. No hard dependency on ``boto3``: the client is injected, so the class is
testable with a fake and the dependency stays optional until the deployment
profile actually selects this backend::

    store = S3ArtifactStore(client, bucket="fiximg")

Remote stores cannot hand out a local path — :meth:`open_path` raises
``NotImplementedError`` on purpose, so callers that need a path must download
explicitly instead of silently depending on a shared filesystem.
"""
from __future__ import annotations

import os
from typing import BinaryIO

from fiximg.domain.artifacts import ArtifactRef
from fiximg.infrastructure.storage.base import BaseArtifactStore


class S3ArtifactStore(BaseArtifactStore):
    """Artifact store backed by an S3-compatible object store."""

    backend = "s3"

    def __init__(self, client, bucket: str, prefix: str = "") -> None:
        if not bucket:
            raise ValueError("S3ArtifactStore requires a bucket name")
        self.client = client
        self.bucket = bucket
        self.prefix = prefix.strip("/")

    # ---------------------------------------------------------------- helpers
    def _object_key(self, key: str) -> str:
        key = self._normalise(key)
        return f"{self.prefix}/{key}" if self.prefix else key

    def _ref(self, key: str) -> ArtifactRef:
        return ArtifactRef(key=self._normalise(key), backend=self.backend)

    # --------------------------------------------------------------- protocol
    def put_bytes(self, data: bytes, key: str, mime_type: str | None = None) -> ArtifactRef:
        extra = {"ContentType": mime_type} if mime_type else None
        self.client.put_object(
            Bucket=self.bucket, Key=self._object_key(key), Body=data, **(extra or {})
        )
        return self._ref(key)

    def put_file(self, source_path: str, key: str, mime_type: str | None = None) -> ArtifactRef:
        with open(source_path, "rb") as handle:
            return self.put_bytes(handle.read(), key, mime_type)

    def get_bytes(self, ref: ArtifactRef) -> bytes:
        try:
            response = self.client.get_object(
                Bucket=self.bucket, Key=self._object_key(ref.key)
            )
        except Exception as exc:  # noqa: BLE001 — normalise backend errors
            raise self._missing(ref) from exc
        return response["Body"].read()

    def open_stream(self, ref: ArtifactRef) -> BinaryIO:
        import io

        return io.BytesIO(self.get_bytes(ref))

    def open_path(self, ref: ArtifactRef) -> str:
        raise NotImplementedError(
            "S3 artifacts have no local path; download the artifact first "
            "(ArtifactStore.get_bytes) or use LocalArtifactStore."
        )

    def delete(self, ref: ArtifactRef) -> None:
        try:
            self.client.delete_object(Bucket=self.bucket, Key=self._object_key(ref.key))
        except Exception:  # noqa: BLE001 — delete is best-effort
            pass

    def exists(self, ref: ArtifactRef) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._object_key(ref.key))
        except Exception:  # noqa: BLE001 — any failure means "not there"
            return False
        return True

    def download_to(self, ref: ArtifactRef, destination: str) -> str:
        """Materialise a remote artifact locally (used by result endpoints)."""
        os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
        with open(destination, "wb") as handle:
            handle.write(self.get_bytes(ref))
        return destination


def build_s3_store_from_settings(settings_obj) -> S3ArtifactStore | None:
    """Create the store from settings; None when S3 is not configured.

    ``FIXIMG_STORAGE_BACKEND=s3`` plus ``FIXIMG_STORAGE_BUCKET`` and
    ``FIXIMG_S3_ENDPOINT`` are required; ``boto3`` must be installed.
    """
    if getattr(settings_obj, "storage_backend", "local") != "s3":
        return None
    bucket = getattr(settings_obj, "storage_bucket", "") or ""
    if not bucket:
        raise ValueError("FIXIMG_STORAGE_BUCKET is required when storage_backend=s3")
    try:
        import boto3  # noqa: PLC0415 — optional dependency
    except ImportError as exc:  # pragma: no cover - depends on deployment
        raise ImportError(
            "boto3 is required for the S3 artifact backend "
            "(pip install 'fiximg[s3]')"
        ) from exc
    endpoint = getattr(settings_obj, "s3_endpoint", "") or None
    region = getattr(settings_obj, "storage_region", "") or None
    access_key = getattr(settings_obj, "storage_access_key", "") or None
    secret_key = getattr(settings_obj, "storage_secret_key", "") or None
    # Empty credentials stay out of the call so boto3's ambient chain (instance
    # role, web identity, ~/.aws) still works; a self-hosted MinIO has no such
    # chain, which is what the two keys and the region are for.
    credentials = ({"aws_access_key_id": access_key, "aws_secret_access_key": secret_key}
                   if access_key and secret_key else {})
    client = boto3.client("s3", endpoint_url=endpoint,
                          region_name=region, **credentials)
    return S3ArtifactStore(client, bucket, prefix=getattr(settings_obj, "storage_prefix", ""))


__all__ = ["S3ArtifactStore", "build_s3_store_from_settings"]
