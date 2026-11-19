from std.python import PythonObject
from std.collections import Array
from views import View
from linalg import multiply

# The degree, coefficients, branch and loop extent are compilation inputs.
def polynomial[degree: Int, unroll: Bool](x: Float64) -> Float64:
    comptime coefficients = (1.0,2.0,3.0,4.0)
    var value: Float64 = 0
    comptime if unroll:
        comptime for k in range(degree+1):
            value = value*x+Float64(coefficients[degree-k])
    else:
        # Same algorithm; coefficient values copied to native runtime storage.
        var c = Array[Float64,4](fill=0)
        comptime for k in range(4):
            c[k] = Float64(coefficients[k])
        for k in range(degree+1):
            value = value*x+c[degree-k]
    return value

def evaluate(x_obj: PythonObject, options: PythonObject) raises -> PythonObject:
    var x = Float64(py=x_obj)
    var degree = Int(py=options[0])
    var unroll = Bool(options[1])
    if degree == 2:
        if unroll:
            return PythonObject(polynomial[2,True](x))
        return PythonObject(polynomial[2,False](x))
    if unroll:
        return PythonObject(polynomial[3,True](x))
    return PythonObject(polynomial[3,False](x))

struct Vec[dtype: DType, N: Int](Movable):
    var data: Array[Scalar[Self.dtype],Self.N]

    def __init__(out self, source: View[Self.dtype]):
        self.data = Array[Scalar[Self.dtype],Self.N](fill=0)
        comptime for i in range(Self.N):
            self.data[i] = source.get(i)

    def dot(self, other: Self) -> Scalar[Self.dtype]:
        var value = Scalar[Self.dtype](0)
        comptime for i in range(Self.N):
            value += self.data[i]*other.data[i]
        return value

def fixed_dot[dtype: DType,N: Int](a_obj: PythonObject,b_obj: PythonObject) raises -> PythonObject:
    var a = View[dtype](a_obj)
    var b = View[dtype](b_obj)
    var x = Vec[dtype,N](a)
    var y = Vec[dtype,N](b)
    return PythonObject(x.dot(y))

def dispatch_dot[dtype: DType](a: PythonObject,b: PythonObject,n: Int) raises -> PythonObject:
    if n == 2:
        return fixed_dot[dtype,2](a,b)
    elif n == 3:
        return fixed_dot[dtype,3](a,b)
    return fixed_dot[dtype,4](a,b)

def generic_dot(a: PythonObject,b: PythonObject) raises -> PythonObject:
    var n = Int(py=a.size)
    if String(py=a.dtype.name) == "float32":
        return dispatch_dot[.float32](a,b,n)
    return dispatch_dot[.float64](a,b,n)
