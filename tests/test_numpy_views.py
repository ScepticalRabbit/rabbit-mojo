import numpy as np
import pytest
from rabbit_mojo.bindings import scale_inplace

@pytest.mark.parametrize('dtype', [np.float64, np.float32, np.int64])
@pytest.mark.parametrize('shape', [(8,), (2,4), (2,2,2), (0,), (2,0,3)])
def test_contiguous(shape, dtype):
    a = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    expected = a.copy()*2
    address = a.ctypes.data
    assert scale_inplace(a) is a
    assert a.ctypes.data == address
    np.testing.assert_array_equal(a, expected)

@pytest.mark.parametrize('view', [lambda a:a[1:], lambda a:a[::2], lambda a:a[::-1],
                                  lambda a:a.reshape(3,4).T, lambda a:a.reshape(3,4)[:,::2]])
def test_strided(view):
    base = np.arange(12.)
    a = view(base)
    expected = a.copy()*3
    scale_inplace(a,3)
    np.testing.assert_array_equal(a,expected)
    assert np.shares_memory(a,base)

def test_safety():
    a = np.arange(4.)
    a.flags.writeable = False
    with pytest.raises(ValueError): scale_inplace(a)
    with pytest.raises(TypeError): scale_inplace(np.arange(4,dtype=np.int32))
    with pytest.raises(ValueError): scale_inplace(np.ndarray((2,),dtype='f8',buffer=bytearray(17),offset=1))
    with pytest.raises(ValueError): scale_inplace(np.lib.stride_tricks.as_strided(np.arange(2.),shape=(20,)))
    with pytest.raises(ValueError): scale_inplace(np.lib.stride_tricks.as_strided(np.arange(2.),shape=(2,2),strides=(0,8)))
    with pytest.raises(OverflowError): scale_inplace(np.array([2**62],dtype=np.int64),4)
