from __future__ import annotations

import json
from typing import Iterable, List, Sequence

from .alerts import Alert
from .models import GPU, GPUProcess


def human_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, _ = divmod(seconds, 60)
    if days:
        return f"{days}d{hours}h"
    if hours:
        return f"{hours}h{minutes}m"
    return f"{minutes}m"


def json_dump(data: object) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


def table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    rows = [[str(cell) for cell in row] for row in rows]
    widths: List[int] = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    line = "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    sep = "  ".join("-" * width for width in widths)
    body = ["  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)) for row in rows]
    return "\n".join([line, sep, *body])


def gpu_rows(gpus: Sequence[GPU], alerts: Sequence[Alert]) -> List[List[object]]:
    alert_by_gpu = {}
    for alert in alerts:
        alert_by_gpu.setdefault(alert.gpu_index, []).append(alert.message)
    return [
        [
            gpu.index,
            gpu.name,
            f"{gpu.memory_used_mb}/{gpu.memory_total_mb} MB",
            f"{gpu.memory_free_mb} MB",
            f"{gpu.utilization_gpu_percent}%",
            "-" if gpu.temperature_c is None else f"{gpu.temperature_c}C",
            "-" if gpu.power_w is None else f"{gpu.power_w:.0f}W",
            "; ".join(alert_by_gpu.get(gpu.index, [])) or "-",
        ]
        for gpu in gpus
    ]


def process_rows(processes: Sequence[GPUProcess]) -> List[List[object]]:
    return [
        [
            proc.gpu_index if proc.gpu_index is not None else "-",
            proc.pid,
            proc.user or "-",
            f"{proc.used_memory_mb} MB",
            human_duration(proc.running_seconds),
            proc.command or proc.process_name,
            proc.cwd or "-",
        ]
        for proc in processes
    ]

