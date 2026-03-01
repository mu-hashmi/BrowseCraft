from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SourceName = Literal["browser_use"]


@dataclass(slots=True)
class CandidateFile:
    source: SourceName
    canonical_url: str
    download_url: str
    filename: str
    title: str
    score: float
    browser_task_id: str | None = None
    browser_output_file_id: str | None = None

    @property
    def extension(self) -> str:
        return Path(self.filename).suffix.lower()
