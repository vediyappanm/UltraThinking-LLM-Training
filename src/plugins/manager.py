"""
Simple plugin manager for adding architectures, datasets, or agents.
"""
from __future__ import annotations

from typing import Dict, Callable


class PluginManager:
    def __init__(self):
        self.registry: Dict[str, Callable] = {}

    def install(self, name: str, factory: Callable):
        self.registry[name] = factory

    def enable(self, name: str):
        if name not in self.registry:
            raise KeyError(f"Plugin {name} not installed")
        return self.registry[name]()

    def available(self):
        return list(self.registry.keys())
