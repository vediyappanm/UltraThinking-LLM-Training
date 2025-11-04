"""
Model Zoo registry for public/private checkpoints.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Any, List


class ModelZoo:
    def __init__(self, root: str = "./model_zoo"):
        self.root = root
        os.makedirs(self.root, exist_ok=True)
        self.manifest_path = os.path.join(self.root, "manifest.json")
        if not os.path.exists(self.manifest_path):
            with open(self.manifest_path, "w") as f:
                json.dump({"models": []}, f)

    def register(self, name: str, path: str, tags: List[str] = None, private: bool = False, metadata: Dict[str, Any] = None):
        data = self._load()
        data["models"].append({"name": name, "path": path, "tags": tags or [], "private": private, "metadata": metadata or {}})
        self._save(data)

    def list(self, include_private: bool = False) -> List[Dict[str, Any]]:
        data = self._load()
        return [m for m in data["models"] if include_private or not m.get("private", False)]

    def _load(self):
        return json.load(open(self.manifest_path, "r"))

    def _save(self, data):
        json.dump(data, open(self.manifest_path, "w"), indent=2)
