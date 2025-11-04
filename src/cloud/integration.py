"""
Cloud integration scaffolding for AWS/GCP/Azure and TPU v5e.

Features:
- Provider detection via environment and SDK availability
- Storage clients: S3 / GCS / Azure Blob
- Cost estimation stubs (heuristics)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class CloudContext:
    provider: str  # aws|gcp|azure|local
    region: Optional[str] = None
    project: Optional[str] = None
    bucket: Optional[str] = None


def detect_provider() -> CloudContext:
    # Simple heuristics based on env vars
    if os.getenv("AWS_EXECUTION_ENV") or os.getenv("AWS_REGION"):
        return CloudContext(provider="aws", region=os.getenv("AWS_REGION"))
    if os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT"):
        return CloudContext(provider="gcp", project=os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT"))
    if os.getenv("AZURE_REGION") or os.getenv("AZURE_TENANT_ID"):
        return CloudContext(provider="azure", region=os.getenv("AZURE_REGION"))
    return CloudContext(provider="local")


class StorageClient:
    def __init__(self, ctx: CloudContext):
        self.ctx = ctx
        self.client = None
        self.kind = None
        self._init()

    def _init(self):
        try:
            if self.ctx.provider == "aws":
                import boto3  # type: ignore
                self.client = boto3.client("s3")
                self.kind = "s3"
            elif self.ctx.provider == "gcp":
                from google.cloud import storage  # type: ignore
                self.client = storage.Client()
                self.kind = "gcs"
            elif self.ctx.provider == "azure":
                from azure.storage.blob import BlobServiceClient  # type: ignore
                conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
                if conn:
                    self.client = BlobServiceClient.from_connection_string(conn)
                self.kind = "blob"
            else:
                self.kind = "local"
        except Exception:
            self.client = None
            self.kind = "local"

    def upload(self, local_path: str, remote_path: str, bucket: Optional[str] = None):
        if self.kind == "s3" and self.client:
            b = bucket or os.getenv("S3_BUCKET") or "ultrathink"
            self.client.upload_file(local_path, b, remote_path)
        elif self.kind == "gcs" and self.client:
            bname = bucket or os.getenv("GCS_BUCKET") or "ultrathink"
            bucket_obj = self.client.bucket(bname)
            blob = bucket_obj.blob(remote_path)
            blob.upload_from_filename(local_path)
        elif self.kind == "blob" and self.client:
            bname = bucket or os.getenv("AZURE_BLOB_CONTAINER") or "ultrathink"
            container = self.client.get_container_client(bname)
            with open(local_path, "rb") as f:
                container.upload_blob(remote_path, f, overwrite=True)
        else:
            # local noop
            pass


def estimate_training_cost(
    num_params: int,
    hours: float,
    hardware: str = "A100",
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Rough cost estimator (heuristic)."""
    # USD/hour estimates (very rough, placeholder)
    rates = {
        "A100": 3.5,
        "H100": 6.0,
        "L4": 0.6,
        "TPU-v5e": 2.0,
    }
    r = rates.get(hardware, 2.0)
    return {
        "hardware": hardware,
        "hours": hours,
        "rate_usd_per_hour": r,
        "estimated_cost_usd": round(r * hours, 2),
        "params_billion": round(num_params / 1e9, 2),
        "provider": provider or detect_provider().provider,
    }


def tpu_available() -> bool:
    # TPU detection stub
    return os.getenv("TPU_NAME") is not None
