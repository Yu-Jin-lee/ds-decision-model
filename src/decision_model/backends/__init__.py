"""Backend registry: the model config's ``backend`` key picks the class."""

from __future__ import annotations

from typing import Any

from .base import Backend


def get_backend(model_config: dict[str, Any]) -> Backend:
    name = model_config["backend"]
    if name == "clef":
        from .clef import ClefBackend

        return ClefBackend(model_config)
    if name == "kev":
        from .kev import KevBackend

        return KevBackend(model_config)
    if name == "laya":
        from .laya import LayaBackend

        return LayaBackend(model_config)
    raise ValueError(f"unknown backend {name!r}")


__all__ = ["Backend", "get_backend"]
