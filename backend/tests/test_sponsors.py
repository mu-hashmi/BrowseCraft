from __future__ import annotations

from browsecraft_backend import sponsors


def test_verify_sponsor_imports_succeeds() -> None:
    sponsors.verify_sponsor_imports()


def test_initialize_laminar_returns_false_without_api_key() -> None:
    assert sponsors.initialize_laminar(None) is False


def test_laminar_span_is_noop_when_uninitialized(monkeypatch) -> None:
    monkeypatch.setattr(sponsors, "_laminar_initialized", False)
    marker = {"visited": False}

    with sponsors.laminar_span("test.span", payload={"key": "value"}):
        marker["visited"] = True

    assert marker["visited"] is True
