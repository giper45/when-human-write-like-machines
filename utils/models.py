from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel


class FilterStepReport(BaseModel):
    filter_name: str
    rows_before: int
    rows_after: int
    rows_removed: int


class FilteringReport(BaseModel):
    dataset_name: str
    rows_initial: int
    steps: List[FilterStepReport] = []
    rows_final: int = 0

    def add_step(self, filter_name: str, rows_before: int, rows_after: int) -> None:
        self.steps.append(
            FilterStepReport(
                filter_name=filter_name,
                rows_before=rows_before,
                rows_after=rows_after,
                rows_removed=rows_before - rows_after,
            )
        )
        self.rows_final = rows_after

    def save(self, output_dir: str | Path) -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        dest = out / "filtering_report.json"
        dest.write_text(self.model_dump_json(indent=2))
        return dest
