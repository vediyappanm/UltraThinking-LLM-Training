"""
Dataset lineage tracking and audit logging utilities.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional


class AuditLogger:
    def __init__(self, path: str = "./audit_log.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def log(self, event: Dict[str, Any]):
        event = {**event, "ts": datetime.utcnow().isoformat()}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")


def register_dataset(name: str, version: str, source: str, checksum: str, extra: Optional[Dict[str, Any]] = None, logger: Optional[AuditLogger] = None):
    evt = {
        "type": "dataset_register",
        "name": name,
        "version": version,
        "source": source,
        "checksum": checksum,
        "extra": extra or {},
    }
    (logger or AuditLogger()).log(evt)


def log_training_run(config: Dict[str, Any], datasets: Dict[str, str], output_path: str, logger: Optional[AuditLogger] = None):
    evt = {
        "type": "training_run",
        "config": config,
        "datasets": datasets,
        "output_path": output_path,
    }
    (logger or AuditLogger()).log(evt)
