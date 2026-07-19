# GPUOps

GPUOps is a small CLI for managing shared lab GPUs. It focuses on the first
things an admin usually needs: seeing who is using GPUs, finding suspicious
idle jobs, safely killing selected GPU processes, and keeping lightweight usage
history.

## Features

- Real-time GPU status: memory, utilization, temperature, power, and processes.
- GPU process list with user, command, working directory, and running time.
- Kill all processes on a GPU, all GPU processes for a user, or that user on one GPU.
- Sorting by free memory, used memory, utilization, temperature, or GPU index.
- Stalled-GPU warning when memory is occupied but utilization stays near zero.
- Per-user summary across all GPUs.
- Memory/running-time top view.
- Lightweight JSONL history sampling and GPU-hour aggregation.
- Safe-by-default process management: `kill` is dry-run unless `--yes` is passed.

## Install locally

```bash
python3 -m pip install -e .
```

## Commands

```bash
gpuops status
gpuops status --sort free --record
gpuops ps --gpu 0
gpuops users
gpuops top --by memory
gpuops doctor
gpuops history --days 30
```

Kill commands are dry-run by default:

```bash
gpuops kill --gpu 3
gpuops kill --user alice
gpuops kill --gpu 3 --user alice
```

To actually send signals:

```bash
gpuops kill --gpu 3 --user alice --yes
gpuops kill --user alice --force --yes
```

## History

`gpuops status --record` appends one sample to:

```text
~/.local/share/gpuops/history.jsonl
```

Run it from cron or systemd every minute if you want durable usage stats.

## Notes

GPUOps uses `nvidia-smi` when available and enriches process information from
`/proc` on Linux. On machines without NVIDIA GPUs, commands fail cleanly with a
message instead of crashing.
