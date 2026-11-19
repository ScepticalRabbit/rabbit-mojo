from pathlib import Path
import subprocess
import sys
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SCRIPTS = sorted(EXAMPLES.glob("ex[0-9][0-9]_*.py"))


def test_all_examples_present():
    assert len(SCRIPTS) == 28
    assert [p.name[2:4] for p in SCRIPTS] == [f"{i:02}" for i in range(1, 29)]


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.stem)
def test_example(script, tmp_path):
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
