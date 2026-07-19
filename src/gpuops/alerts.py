from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .models import GPU, Snapshot


@dataclass(frozen=True)
class Alert:
    gpu_index: int
    level: str
    message: str


def evaluate_snapshot(
    snapshot: Snapshot,
    idle_util_percent: int = 5,
    occupied_memory_mb: int = 1024,
    high_temp_c: int = 82,
) -> List[Alert]:
    alerts: List[Alert] = []
    process_by_gpu = {}
    for proc in snapshot.processes:
        if proc.gpu_index is not None:
            process_by_gpu.setdefault(proc.gpu_index, 0)
            process_by_gpu[proc.gpu_index] += 1

    for gpu in snapshot.gpus:
        alerts.extend(_gpu_alerts(gpu, process_by_gpu.get(gpu.index, 0), idle_util_percent, occupied_memory_mb, high_temp_c))
    return alerts


def _gpu_alerts(
    gpu: GPU,
    process_count: int,
    idle_util_percent: int,
    occupied_memory_mb: int,
    high_temp_c: int,
) -> List[Alert]:
    alerts: List[Alert] = []
    if gpu.memory_used_mb >= occupied_memory_mb and gpu.utilization_gpu_percent <= idle_util_percent:
        alerts.append(
            Alert(
                gpu_index=gpu.index,
                level="warn",
                message="memory occupied while utilization is near zero",
            )
        )
    if gpu.memory_used_mb >= occupied_memory_mb and process_count == 0:
        alerts.append(
            Alert(
                gpu_index=gpu.index,
                level="warn",
                message="memory occupied but nvidia-smi reports no compute process",
            )
        )
    if gpu.temperature_c is not None and gpu.temperature_c >= high_temp_c:
        alerts.append(Alert(gpu_index=gpu.index, level="crit", message=f"temperature is high ({gpu.temperature_c}C)"))
    return alerts

