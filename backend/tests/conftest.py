from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--quick",
        action="store_true",
        default=False,
        help="Run only the quick_spatial subset when spatial tests are selected.",
    )
    parser.addoption(
        "--with-planning",
        action="store_true",
        default=False,
        help="Include live spatial tests that intentionally exercise planning or preview flows.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        if not config.getoption("--with-planning") and "planner_spatial" in item.keywords:
            deselected.append(item)
            continue
        if config.getoption("--quick") and "spatial" in item.keywords and "quick_spatial" not in item.keywords:
            deselected.append(item)
            continue
        selected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected
