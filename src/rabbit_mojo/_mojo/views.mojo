"""Synchronous borrowed pointer + shape + signed element strides; no Python in loops."""
from std.python import PythonObject
from std.python.numpy import from_numpy_tensor

struct View[dtype: DType](Movable):
    var owner: PythonObject
    var data: Pointer[Scalar[Self.dtype], MutUntrackedOrigin]
    var shape: List[Int]
    var strides: List[Int]
    var size: Int

    def __init__(out self, array: PythonObject) raises:
        self.owner = array
        self.data = Pointer[Scalar[Self.dtype], MutUntrackedOrigin](unsafe_from_address=Int(py=array.ctypes.data))
        self.shape = List[Int]()
        self.strides = List[Int]()
        self.size = Int(py=array.size)
        var itemsize = Int(py=array.itemsize)
        for i in range(Int(py=array.ndim)):
            self.shape.append(Int(py=array.shape[i]))
            self.strides.append(Int(py=array.strides[i]) // itemsize)

    def offset(self, index: Int) -> Int:
        var remainder = index
        var result = 0
        for j in range(len(self.shape) - 1, -1, -1):
            result += (remainder % self.shape[j]) * self.strides[j]
            remainder //= self.shape[j]
        return result

    def get(self, index: Int) -> Scalar[Self.dtype]:
        return self.data[unsafe_offset=self.offset(index)]

    def put(self, index: Int, value: Scalar[Self.dtype]):
        self.data[unsafe_offset=self.offset(index)] = value

# Official NumPyView API path, specialised for rank and element type.
def scale_contiguous[dtype: DType, rank: Int](array: PythonObject, factor: Scalar[dtype]) raises:
    var owner = array
    var view = from_numpy_tensor[dtype, rank](owner)
    for i in range(len(view.data)):
        view.data[i] *= factor

def scale_typed[dtype: DType](array: PythonObject, factor: Scalar[dtype], contiguous: Bool) raises:
    var rank = Int(py=array.ndim)
    if contiguous and rank == 1:
        scale_contiguous[dtype, 1](array, factor)
    elif contiguous and rank == 2:
        scale_contiguous[dtype, 2](array, factor)
    elif contiguous and rank == 3:
        scale_contiguous[dtype, 3](array, factor)
    else:
        var view = View[dtype](array)
        for i in range(view.size):
            view.put(i, view.get(i) * factor)

def scale(array: PythonObject, factor: PythonObject) raises -> PythonObject:
    var f = Float64(py=factor)
    var contiguous = Bool(array.flags.c_contiguous)
    var dtype = String(py=array.dtype.name)
    if dtype == "float64":
        scale_typed[.float64](array, f, contiguous)
    elif dtype == "float32":
        scale_typed[.float32](array, Float32(f), contiguous)
    else:
        scale_typed[.int64](array, Int64(Int(py=factor)), contiguous)
    return array
