from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SeizureInterval:
    start_sec: float
    end_sec: float

    def to_dict(self) -> dict[str, float]:
        return {"start_sec": self.start_sec, "end_sec": self.end_sec}


_FILE_RE = re.compile(r"File Name:\s*(?P<name>\S+)", re.IGNORECASE)
_SEIZURE_COUNT_RE = re.compile(r"Number of Seizures in File:\s*(?P<count>\d+)", re.IGNORECASE)
_START_RE = re.compile(r"Seizure(?: \d+)? Start Time:\s*(?P<time>[\d.]+)\s*seconds", re.IGNORECASE)
_END_RE = re.compile(r"Seizure(?: \d+)? End Time:\s*(?P<time>[\d.]+)\s*seconds", re.IGNORECASE)


def parse_summary_file(path: str | Path) -> dict[str, list[SeizureInterval]]:
    """Parse a CHB-MIT `chbXX-summary.txt` file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")

    annotations: dict[str, list[SeizureInterval]] = {}
    current_file: str | None = None
    starts: list[float] = []
    ends: list[float] = []

    def flush() -> None:
        nonlocal starts, ends
        if current_file is None:
            return
        if len(starts) != len(ends):
            raise ValueError(f"Mismatched seizure start/end entries for {current_file} in {path}")
        annotations[current_file] = [SeizureInterval(s, e) for s, e in zip(starts, ends)]
        starts, ends = [], []

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        file_match = _FILE_RE.search(line)
        if file_match:
            flush()
            current_file = file_match.group("name")
            annotations.setdefault(current_file, [])
            continue
        if current_file is None:
            continue
        if _SEIZURE_COUNT_RE.search(line):
            continue
        start_match = _START_RE.search(line)
        if start_match:
            starts.append(float(start_match.group("time")))
            continue
        end_match = _END_RE.search(line)
        if end_match:
            ends.append(float(end_match.group("time")))
    flush()
    return annotations


def intervals_to_json(intervals: list[SeizureInterval]) -> list[dict[str, float]]:
    return [interval.to_dict() for interval in intervals]


def intervals_overlap(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
    min_overlap_fraction: float = 1.0,
) -> bool:
    """Return true when interval A overlaps B by the requested fraction of A."""
    overlap = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    duration = max(end_a - start_a, 1e-12)
    return overlap / duration >= min_overlap_fraction
