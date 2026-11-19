"""Reproducible end-to-end timings; validation and allocation stay inside timed calls."""

import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits
import rabbit_mojo as rm


def measure(function, repeats=9):
    for _ in range(3):
        function()
    values = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        values.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "iqr_ms": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "repeats": repeats,
    }


def report(name, runner):
    with threadpool_limits(limits=1):
        data = runner()
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            "unknown",
        )
        compiler = Path(sys.executable).parent / "mojo"
        version = subprocess.check_output(
            [str(compiler), "--version"], text=True
        ).strip()
        result = {
            "benchmark": name,
            "metadata": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "rabbit_mojo": rm.__version__,
                "mojo": version,
                "platform": platform.platform(),
                "cpu": cpu,
                "logical_cpus": os.cpu_count(),
                "flags": "-O3 --target-cpu x86-64",
                "numpy_thread_limit": 1,
                "mojo_threads": "1 sequential/SIMD; 2 pthread workers where labelled",
                "threadpools": threadpool_info(),
            },
            "measurements": data,
        }
    print(json.dumps(result, indent=2))
