from __future__ import annotations
from dataclasses import dataclass, asdict, is_dataclass
from pathlib import Path
import csv


@dataclass
class MetricsRow:
    frame: int
    t: float
    scenario: str
    model: str
    iou: float
    fp: int
    fn: int
    n_points: int


class CSVLogger:

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._fh = self.path.open("w", newline="", encoding="utf-8")
        self._writer: csv.DictWriter | None = None
        self._header_written = False

    def log(self, row: MetricsRow | dict | None) -> None:
        if row is None:
            return

        if is_dataclass(row):
            data = asdict(row)
        elif isinstance(row, dict):
            data = row
        else:
            raise TypeError("log.")

        if not self._header_written:
            fieldnames = list(data.keys())
            self._writer = csv.DictWriter(self._fh, fieldnames=fieldnames)
            self._writer.writeheader()
            self._header_written = True

        assert self._writer is not None
        self._writer.writerow(data)
        self._fh.flush()

    def close(self) -> None:
        try:
            if self._fh:
                self._fh.close()
        except Exception:
            pass