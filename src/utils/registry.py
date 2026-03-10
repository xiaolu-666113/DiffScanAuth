"""Lightweight registries for model and lightning modules."""

from __future__ import annotations

from typing import Callable

MODEL_REGISTRY: dict[str, Callable] = {}
LIT_REGISTRY: dict[str, Callable] = {}


def register_model(name: str):
    """Decorator to register model class by name."""

    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def register_lit(name: str):
    """Decorator to register lightning module class by name."""

    def decorator(cls):
        LIT_REGISTRY[name] = cls
        return cls

    return decorator
