import logging

import uvicorn

from .config import get_settings


def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    settings = get_settings()
    uvicorn.run(
        "browsecraft_backend.app:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
        log_level="info",
    )
