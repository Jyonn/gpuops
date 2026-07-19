from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .models import Snapshot


DEFAULT_HISTORY_PATH = Path(os.environ.get("GPUOPS_HISTORY", "~/.local/share/gpuops/history.jsonl")).expanduser()


@dataclass(frozen=True)
class UserUsage:
    user: str
    gpu_hours: float
    memory_gb_hours: float
    samples: int


def record_snapshot(snapshot: Snapshot, path: Path = DEFAULT_HISTORY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(snapshot.to_dict(), sort_keys=True) + "\n")


def read_snapshots(path: Path = DEFAULT_HISTORY_PATH, since: Optional[float] = None) -> List[dict]:
    if not path.exists():
        return []
    samples: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            if since is None or float(sample.get("timestamp", 0)) >= since:
                samples.append(sample)
    samples.sort(key=lambda item: float(item.get("timestamp", 0)))
    return samples


def summarize_user_usage(samples: Iterable[dict], max_interval_seconds: int = 300) -> List[UserUsage]:
    ordered = sorted(samples, key=lambda item: float(item.get("timestamp", 0)))
    accum: Dict[str, Dict[str, float]] = {}
    for current, nxt in zip(ordered, ordered[1:]):
        timestamp = float(current.get("timestamp", 0))
        next_timestamp = float(nxt.get("timestamp", timestamp))
        interval = max(0.0, min(max_interval_seconds, next_timestamp - timestamp))
        if interval <= 0:
            continue
        seen_user_gpu = set()
        for proc in current.get("processes", []):
            user = proc.get("user") or "unknown"
            gpu_index = proc.get("gpu_index")
            used_memory_mb = float(proc.get("used_memory_mb") or 0)
            bucket = accum.setdefault(user, {"gpu_seconds": 0.0, "memory_mb_seconds": 0.0, "samples": 0.0})
            key = (user, gpu_index)
            if key not in seen_user_gpu:
                bucket["gpu_seconds"] += interval
                seen_user_gpu.add(key)
            bucket["memory_mb_seconds"] += used_memory_mb * interval
            bucket["samples"] += 1

    return sorted(
        [
            UserUsage(
                user=user,
                gpu_hours=values["gpu_seconds"] / 3600.0,
                memory_gb_hours=values["memory_mb_seconds"] / 1024.0 / 3600.0,
                samples=int(values["samples"]),
            )
            for user, values in accum.items()
        ],
        key=lambda item: item.gpu_hours,
        reverse=True,
    )


def since_days(days: int) -> float:
    return time.time() - (days * 86400)
