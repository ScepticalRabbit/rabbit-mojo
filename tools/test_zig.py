"""Optional independent Zig tests, then numerical comparison with native Mojo."""

from pathlib import Path
import re
import shutil
import subprocess
import numpy as np
import rabbit_mojo as rm


def main():
    zig = shutil.which("zig")
    if not zig:
        print("Zig unavailable: optional comparisons untested")
        return
    files = sorted(
        (Path(__file__).resolve().parents[1] / "examples" / "zig").glob("ex*.zig")
    )
    if len(files) != 4:
        raise RuntimeError("Expected four Zig pairs")
    for path in files:
        result = subprocess.run(
            [zig, "test", str(path)], capture_output=True, text=True, check=True
        )
        log = result.stdout + result.stderr
        number = int(path.name[2:4])
        if number == 25:
            match = re.search(r"quadratic=([\d.]+) cubic=([\d.]+)", log)
            if not match:
                raise AssertionError(log)
            np.testing.assert_allclose(
                [float(x) for x in match.groups()],
                [rm.polynomial(2.0, degree=d) for d in (2, 3)],
                rtol=0,
                atol=1e-13,
            )
        elif number == 26:
            rows = re.findall(r"(f32|f64) N=(\d+) dot=([\d.]+)", log)
            assert len(rows) == 6, log
            for dtype, n, value in rows:
                dtype = np.float32 if dtype == "f32" else np.float64
                n = int(n)
                a = np.arange(1, n + 1, dtype=dtype)
                b = np.arange(n, 0, -1, dtype=dtype)
                np.testing.assert_allclose(
                    float(value), rm.generic_dot(a, b), rtol=1e-6, atol=1e-13
                )
        elif number == 27:
            rows = re.findall(r"N=(\d+) product=\{([^}]+)\}", log)
            assert len(rows) == 2, log
            for n, values in rows:
                n = int(n)
                a = np.arange(1, n * n + 1, dtype=float).reshape(n, n)
                np.testing.assert_allclose(
                    np.fromstring(values, sep=","),
                    rm.matmul(a, np.eye(n) * 2).ravel(),
                    rtol=0,
                    atol=1e-13,
                )
        else:
            rows = re.findall(
                r"x=([-\d.]+) unrolled=([-\d.]+) runtime_loop=([-\d.]+)", log
            )
            assert len(rows) == 3, log
            for x, unrolled, loop in rows:
                np.testing.assert_allclose(
                    [float(unrolled), float(loop)],
                    [
                        rm.polynomial(float(x), degree=3),
                        rm.polynomial(float(x), degree=3, unroll=False),
                    ],
                    rtol=0,
                    atol=1e-13,
                )
        print("PASS", path.name, "independent Zig test and Mojo numerical agreement")


if __name__ == "__main__":
    main()
