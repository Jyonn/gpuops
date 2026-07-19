import unittest

from gpuops.nvidia import collect_snapshot


class NvidiaParsingTests(unittest.TestCase):
    def test_collect_snapshot_maps_processes_to_gpu_index(self):
        def runner(args):
            query = args[0]
            if query.startswith("--query-gpu"):
                return "0, GPU-a, NVIDIA A100, 81920, 40960, 0, 70, 250.5\n1, GPU-b, NVIDIA A100, 81920, 0, 0, 35, [N/A]\n"
            if query.startswith("--query-compute-apps"):
                return "123, GPU-a, 20480, python\n"
            raise AssertionError(args)

        snapshot = collect_snapshot(runner)

        self.assertEqual(len(snapshot.gpus), 2)
        self.assertEqual(snapshot.gpus[0].memory_free_mb, 40960)
        self.assertEqual(snapshot.processes[0].gpu_index, 0)
        self.assertEqual(snapshot.processes[0].used_memory_mb, 20480)


if __name__ == "__main__":
    unittest.main()
