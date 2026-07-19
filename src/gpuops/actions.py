from __future__ import annotations

import os
import signal
from dataclasses import dataclass
from typing import Iterable, List, Optional

from .models import GPUProcess


@dataclass(frozen=True)
class KillResult:
    pid: int
    user: Optional[str]
    gpu_index: Optional[int]
    signal_name: str
    killed: bool
    message: str


def filter_processes(processes: Iterable[GPUProcess], gpu: Optional[int] = None, user: Optional[str] = None) -> List[GPUProcess]:
    selected = []
    for proc in processes:
        if gpu is not None and proc.gpu_index != gpu:
            continue
        if user is not None and proc.user != user:
            continue
        selected.append(proc)
    return selected


def signal_for(force: bool) -> signal.Signals:
    return signal.SIGKILL if force else signal.SIGTERM


def kill_processes(processes: Iterable[GPUProcess], sig: signal.Signals, dry_run: bool = True) -> List[KillResult]:
    results: List[KillResult] = []
    for proc in processes:
        if dry_run:
            results.append(
                KillResult(
                    pid=proc.pid,
                    user=proc.user,
                    gpu_index=proc.gpu_index,
                    signal_name=sig.name,
                    killed=False,
                    message="dry-run",
                )
            )
            continue
        try:
            os.kill(proc.pid, sig)
            results.append(
                KillResult(
                    pid=proc.pid,
                    user=proc.user,
                    gpu_index=proc.gpu_index,
                    signal_name=sig.name,
                    killed=True,
                    message="signal sent",
                )
            )
        except ProcessLookupError:
            results.append(
                KillResult(
                    pid=proc.pid,
                    user=proc.user,
                    gpu_index=proc.gpu_index,
                    signal_name=sig.name,
                    killed=False,
                    message="process already exited",
                )
            )
        except PermissionError:
            results.append(
                KillResult(
                    pid=proc.pid,
                    user=proc.user,
                    gpu_index=proc.gpu_index,
                    signal_name=sig.name,
                    killed=False,
                    message="permission denied",
                )
            )
    return results

