from __future__ import annotations

from contextlib import contextmanager
from typing import Any

_laminar_initialized = False


def initialize_laminar(api_key: str | None) -> bool:
    global _laminar_initialized
    if not api_key:
        return False
    if _laminar_initialized:
        return True

    from lmnr import Instruments, Laminar

    Laminar.initialize(
        project_api_key=api_key,
        disabled_instruments={
            Instruments.ANTHROPIC,
        },
    )
    _laminar_initialized = True
    return True


@contextmanager
def laminar_span(name: str, payload: dict[str, Any] | None = None):
    if not _laminar_initialized:
        yield
        return

    from lmnr import Laminar

    with Laminar.start_as_current_span(name=name, input=payload):
        yield


def verify_sponsor_imports() -> None:
    from anthropic import Anthropic
    from supermemory import Supermemory

    _ = Anthropic
    _ = Supermemory
