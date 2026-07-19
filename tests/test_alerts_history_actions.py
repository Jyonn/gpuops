import signal
import unittest

from gpuops.actions import filter_processes, kill_processes
from gpuops.alerts import evaluate_snapshot
from gpuops.history import summarize_user_usage
from gpuops.models import GPU, GPUProcess, Snapshot


class AlertsHistoryActionsTests(unittest.TestCase):
    def test_alerts_idle_occupied_gpu(self):
        snapshot = Snapshot(
            timestamp=1,
            gpus=[GPU(0, "GPU-a", "A100", 80000, 70000, 0, 74, 220.0)],
            processes=[GPUProcess(10, "GPU-a", 0, 70000, "python", user="alice")],
        )

        alerts = evaluate_snapshot(snapshot)

        self.assertEqual(alerts[0].gpu_index, 0)
        self.assertIn("near zero", alerts[0].message)

    def test_history_counts_gpu_hours_once_per_user_gpu(self):
        samples = [
            {
                "timestamp": 0,
                "processes": [
                    {"user": "alice", "gpu_index": 0, "used_memory_mb": 1024},
                    {"user": "alice", "gpu_index": 0, "used_memory_mb": 2048},
                    {"user": "bob", "gpu_index": 1, "used_memory_mb": 1024},
                ],
            },
            {"timestamp": 3600, "processes": []},
        ]

        usage = summarize_user_usage(samples, max_interval_seconds=3600)

        self.assertEqual(usage[0].user, "alice")
        self.assertAlmostEqual(usage[0].gpu_hours, 1.0)
        self.assertAlmostEqual(usage[0].memory_gb_hours, 3.0)

    def test_filter_and_dry_run_kill(self):
        processes = [
            GPUProcess(10, "GPU-a", 0, 100, "python", user="alice"),
            GPUProcess(11, "GPU-b", 1, 100, "python", user="bob"),
        ]

        selected = filter_processes(processes, gpu=0, user="alice")
        results = kill_processes(selected, signal.SIGTERM, dry_run=True)

        self.assertEqual([proc.pid for proc in selected], [10])
        self.assertFalse(results[0].killed)
        self.assertEqual(results[0].message, "dry-run")


if __name__ == "__main__":
    unittest.main()
