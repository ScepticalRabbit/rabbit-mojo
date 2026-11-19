"""Run and verify every required example in numeric order, from any working directory."""

from pathlib import Path
import subprocess
import sys


def main():
    directory = Path(__file__).resolve().parent
    scripts = sorted(directory.glob("ex[0-9][0-9]_*.py"))
    if len(scripts) != 28 or [p.name[2:4] for p in scripts] != [
        f"{i:02}" for i in range(1, 29)
    ]:
        raise RuntimeError("Expected exactly one example for every number 01..28")
    for script in scripts:
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=directory.parent,
        )
        if result.returncode:
            sys.stderr.write(result.stdout + result.stderr)
            raise RuntimeError(f"{script.name} failed ({result.returncode})")
        print("PASS", script.name)


if __name__ == "__main__":
    main()
