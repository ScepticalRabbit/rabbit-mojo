"""Run correctness-checked benchmarks and render recorded medians and spread."""

import json
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    reports = []
    for name in ("mesh_transform", "tensor", "image_sampling"):
        result = subprocess.run(
            [sys.executable, str(root / "benchmarks" / f"bench_{name}.py")],
            capture_output=True,
            text=True,
            check=True,
        )
        reports.append(json.loads(result.stdout))
    (root / "docs" / "benchmark_results.json").write_text(
        json.dumps(reports, indent=2) + "\n"
    )
    lines = [
        "# Recorded benchmark results",
        "",
        "Measured end-to-end on 19 November 2026. Three warmups and nine independent timed calls per case. Values are milliseconds; spread is interquartile range. Array creation for input data is outside timings. Public API validation, result allocation and native computation are inside timings. No performance claims beyond these local measurements.",
        "",
        "```json",
        json.dumps(reports[0]["metadata"], indent=2),
        "```",
        "",
        "| Workload | N | Implementation | Median ms | IQR ms |",
        "|---|---:|---|---:|---:|",
    ]
    for report in reports:
        for row in report["measurements"]:
            lines.append(
                f"| {report['benchmark']} | {row['n']} | {row['case']} | {row['median_ms']:.6f} | {row['iqr_ms']:.6f} |"
            )
    lines += [
        "",
        "The raw JSON includes min/max and threadpool details. NumPy uses a threadpool limit of one; native parallel transforms use two freshly created POSIX threads. The scalar source may still be auto-vectorised by the compiler. Explicit SIMD here packs a point and row into four lanes, rather than four separate points. General stride resolution adds overhead. The pthread transform uses direct row/column strides and differs from the general scalar/SIMD loop; compare pthread_1 to parallel_2 to hold that kernel structure constant when inspecting thread effects. Jacobi also computes eigenvectors even when the Python caller requests only values; NumPy eigvalsh computes values only. These differences matter when interpreting timings.",
        "",
        "The raw binding case omits Python validation; it is diagnostic and private, not a supported way to bypass checks. Conversion rows separately measure an explicit float32→float64 allocation/copy and its combined end-to-end cost. Small/large cases show that public boundary costs can dominate small workloads. Thread startup and joins are included. Neither compile-time unrolling nor SIMD is claimed to improve performance without measurements.",
    ]
    (root / "docs" / "benchmark_results.md").write_text("\n".join(lines) + "\n")
    print("Wrote docs/benchmark_results.md and docs/benchmark_results.json")


if __name__ == "__main__":
    main()
