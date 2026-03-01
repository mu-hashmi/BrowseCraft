from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--quick",
        action="store_true",
        default=False,
        help="Run only the quick_spatial subset when spatial tests are selected.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not config.getoption("--quick"):
        return

    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        if "spatial" in item.keywords and "quick_spatial" not in item.keywords:
            deselected.append(item)
            continue
        selected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected
