from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class GPU:
    index: int
    uuid: str
    name: str
    memory_total_mb: int
    memory_used_mb: int
    utilization_gpu_percent: int
    temperature_c: Optional[int]
    power_w: Optional[float]

    @property
    def memory_free_mb(self) -> int:
        return max(0, self.memory_total_mb - self.memory_used_mb)


@dataclass
class GPUProcess:
    pid: int
    gpu_uuid: str
    gpu_index: Optional[int]
    used_memory_mb: int
    process_name: str
    user: Optional[str] = None
    command: Optional[str] = None
    cwd: Optional[str] = None
    started_at: Optional[float] = None
    running_seconds: Optional[float] = None


@dataclass
class Snapshot:
    timestamp: float
    gpus: List[GPU] = field(default_factory=list)
    processes: List[GPUProcess] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "gpus": [gpu.__dict__ for gpu in self.gpus],
            "processes": [proc.__dict__ for proc in self.processes],
        }

