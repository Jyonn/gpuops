from __future__ import annotations

import csv
import subprocess
import time
from io import StringIO
from typing import Callable, Iterable, List, Optional

from .models import GPU, GPUProcess, Snapshot


class NvidiaSmiError(RuntimeError):
    pass


Runner = Callable[[List[str]], str]


def _default_runner(args: List[str]) -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", *args],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise NvidiaSmiError("nvidia-smi was not found on this machine") from exc
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise NvidiaSmiError(f"nvidia-smi failed: {message}") from exc
    return result.stdout


def _rows(text: str) -> Iterable[List[str]]:
    reader = csv.reader(StringIO(text))
    for row in reader:
        cleaned = [cell.strip() for cell in row]
        if cleaned and any(cell for cell in cleaned):
            yield cleaned


def _int(value: str, default: int = 0) -> int:
    value = value.strip()
    if not value or value.upper() == "[N/A]" or value == "N/A":
        return default
    return int(float(value))


def _float_or_none(value: str) -> Optional[float]:
    value = value.strip()
    if not value or value.upper() == "[N/A]" or value == "N/A":
        return None
    return float(value)


def query_gpus(runner: Runner = _default_runner) -> List[GPU]:
    output = runner(
        [
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus: List[GPU] = []
    for row in _rows(output):
        if len(row) < 8:
            continue
        gpus.append(
            GPU(
                index=_int(row[0]),
                uuid=row[1],
                name=row[2],
                memory_total_mb=_int(row[3]),
                memory_used_mb=_int(row[4]),
                utilization_gpu_percent=_int(row[5]),
                temperature_c=None if row[6] in {"", "N/A", "[N/A]"} else _int(row[6]),
                power_w=_float_or_none(row[7]),
            )
        )
    return gpus


def query_processes(runner: Runner = _default_runner) -> List[GPUProcess]:
    try:
        output = runner(
            [
                "--query-compute-apps=pid,gpu_uuid,used_gpu_memory,process_name",
                "--format=csv,noheader,nounits",
            ]
        )
    except NvidiaSmiError as exc:
        text = str(exc).lower()
        if "not supported" in text or "no running processes" in text:
            return []
        raise

    processes: List[GPUProcess] = []
    for row in _rows(output):
        if len(row) < 4 or row[0] in {"", "N/A", "[Not Supported]"}:
            continue
        processes.append(
            GPUProcess(
                pid=_int(row[0]),
                gpu_uuid=row[1],
                gpu_index=None,
                used_memory_mb=_int(row[2]),
                process_name=row[3],
            )
        )
    return processes


def collect_snapshot(runner: Runner = _default_runner) -> Snapshot:
    gpus = query_gpus(runner)
    uuid_to_index = {gpu.uuid: gpu.index for gpu in gpus}
    processes = query_processes(runner)
    for proc in processes:
        proc.gpu_index = uuid_to_index.get(proc.gpu_uuid)
    return Snapshot(timestamp=time.time(), gpus=gpus, processes=processes)

