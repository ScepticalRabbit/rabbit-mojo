"""Point Mojo's Python loader at the already-running interpreter, then restore env.

Mojo 1.2's loader otherwise discovers /usr/bin/python3 and mutates the C process
PYTHONEXECUTABLE/PYTHONPATH environment (invisible to Python's cached os.environ).
"""

import os
from pathlib import Path
import sys
import sysconfig


def load_native():
    keys = ("MOJO_PYTHON_LIBRARY", "PYTHONEXECUTABLE", "PYTHONPATH")
    previous = {key: os.environ.get(key) for key in keys}
    library = Path(sysconfig.get_config_var("LIBDIR")) / sysconfig.get_config_var(
        "LDLIBRARY"
    )
    if not library.is_file():
        raise ImportError(f"Cannot locate running CPython shared library: {library}")
    try:
        os.environ["MOJO_PYTHON_LIBRARY"] = str(library)
        os.environ["PYTHONEXECUTABLE"] = sys.executable
        from . import _native

        return _native
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
                os.unsetenv(key)
            else:
                os.environ[key] = value
