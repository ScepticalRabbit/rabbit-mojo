import gc
import numpy as np
import pytest
import rabbit_mojo as rm


def test_scalars():
    result = rm.scalar_roundtrip(-(2**31), 1 + 2**-30, True, "λ🐇", 3.0)
    assert result[:2] == (-(2**31), -(2**31))
    assert result[2] == 1 + 2**-30 and result[3] == 1.0
    assert result[4:] == (False, "λ🐇 🐇", 6.0)
    assert rm.scalar_roundtrip(2**31 - 1, 0.0, False, "", None)[-1] is None
    with pytest.raises(OverflowError):
        rm.scalar_roundtrip(2**31, 0.0, True, "x")
    with pytest.raises(TypeError):
        rm.scalar_roundtrip(0, 0.0, 0, "x")
    with pytest.raises(TypeError):
        rm.scalar_roundtrip(0, 0.0, True, "x", [])


def test_collections():
    settings = {"scale": 2.0}
    total, coord, mode, wrapped, converted = rm.collection_roundtrip(
        [1.0, 2.0], (3.0, 4.0), settings, rm.Mode.SCALE
    )
    assert (total, coord, mode) == (6.0, 7.0, 1)
    assert wrapped is settings
    assert converted == ([1.0, 2.0], {"scale": 2.0}, (1, "native record"))
    with pytest.raises(TypeError):
        rm.collection_roundtrip(["bad"], (1, 2), settings)
    with pytest.raises(TypeError):
        rm.collection_roundtrip([1], (1, 2), settings, 1)
    with pytest.raises(TypeError):
        rm.collection_roundtrip([1], (1, 2), {})


def test_native_struct_and_dataclass():
    c = rm.Counter(3)
    assert c.increment() == 4
    gc.collect()
    assert c.value() == 4
    with pytest.raises(Exception):
        rm.Counter()
    with pytest.raises(Exception):
        rm.Counter(count=3)
    with pytest.raises(Exception):
        rm.Counter(2**63 - 1).increment()
    config = rm.IsotropicElastic(200.0, 0.25)
    assert rm.lame_parameters(config) == (80.0, 80.0)
    config.youngs_modulus = -1
    with pytest.raises(ValueError):
        rm.lame_parameters(config)
    with pytest.raises(ValueError):
        rm.IsotropicElastic(2.0, 0.5)


def test_native_error_and_kwargs():
    with pytest.raises(Exception, match="useful message"):
        rm.native_error("useful message")
    # Public Python wrappers support keywords. def_function is positional-only here.
    with pytest.raises(TypeError):
        rm._native.scale(array=np.arange(3.0), factor=2.0)


def test_native_keywords_static_and_int64_precision():
    assert rm.native_affine(3.0, factor=2.0, offset=1.0) == 7.0
    assert rm.native_affine(3.0) == 6.0
    with pytest.raises(Exception):
        rm.native_affine(3.0, bad=1.0)
    assert rm.Counter.zero().value() == 0
    with pytest.raises(Exception):
        rm.Counter(3, bad=1)
    a = np.array([1], dtype=np.int64)
    rm.scale_inplace(a, 2**53 + 1)
    assert int(a[0]) == 2**53 + 1


def test_dataclass_roundtrip():
    material = rm.IsotropicElastic(200.0, 0.3)
    converted = rm.material_roundtrip(material)
    assert converted == material and converted is not material


def test_child_interpreter_environment():
    import subprocess
    import sys

    # A native loader may mutate libc environ without updating os.environ.
    # Passing env explicitly would conceal that bug, so inherit it here.
    result = subprocess.run(
        [sys.executable, "-c", "import sys,numpy; print(sys.prefix)"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == sys.prefix
