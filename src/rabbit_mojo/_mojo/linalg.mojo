from std.python import PythonObject
from std.math import sqrt, abs, max
from views import View

def dot(a_obj: PythonObject, b_obj: PythonObject, simd_obj: PythonObject) raises -> PythonObject:
    var a = View[.float64](a_obj)
    var b = View[.float64](b_obj)
    var total: Float64 = 0
    var i = 0
    if Bool(simd_obj):
        var lanes = SIMD[.float64, 4](0)
        while i + 4 <= a.size:
            lanes += a.data.unsafe_offset(i).unsafe_load[width=4]() * b.data.unsafe_offset(i).unsafe_load[width=4]()
            i += 4
        total = lanes.reduce_add()
    while i < a.size:
        total += a.get(i) * b.get(i)
        i += 1
    return PythonObject(total)

def vector(a_obj: PythonObject, b_obj: PythonObject, out_obj: PythonObject, op_obj: PythonObject, factor_obj: PythonObject) raises -> PythonObject:
    var a = View[.float64](a_obj)
    var b = View[.float64](b_obj)
    var dst = View[.float64](out_obj)
    var op = Int(py=op_obj)
    var factor = Float64(py=factor_obj)
    # A native fixed-size value; source may be strided.
    var x = SIMD[.float64, 4](0)
    var y = SIMD[.float64, 4](0)
    for i in range(a.size):
        x[i] = a.get(i)
        y[i] = b.get(i)
    if op == 0:
        x += y
    elif op == 1:
        x *= factor
    elif op == 2:
        var norm = sqrt((x * x).reduce_add())
        if norm == 0:
            raise Error("cannot normalise a zero vector")
        x /= norm
    else:
        x = SIMD[.float64, 4](x[1]*y[2]-x[2]*y[1], x[2]*y[0]-x[0]*y[2], x[0]*y[1]-x[1]*y[0], 0)
    for i in range(a.size):
        dst.put(i, x[i])
    return out_obj

def multiply[N: Int](a: View[.float64], b: View[.float64], dst: View[.float64]):
    comptime for i in range(N):
        comptime for j in range(N):
            var value: Float64 = 0
            comptime for k in range(N):
                value += a.get(i*N+k)*b.get(k*N+j)
            dst.put(i*N+j,value)

def matrix(a_obj: PythonObject, b_obj: PythonObject, out_obj: PythonObject, op_obj: PythonObject) raises -> PythonObject:
    var a = View[.float64](a_obj)
    var b = View[.float64](b_obj)
    var dst = View[.float64](out_obj)
    var n = a.shape[0]
    var op = Int(py=op_obj)
    if op == 0:
        if n == 2:
            multiply[2](a,b,dst)
        else:
            multiply[3](a,b,dst)
    elif op == 1:
        for i in range(n):
            for j in range(n):
                dst.put(j*n+i,a.get(i*n+j))
    elif op == 2:
        for i in range(n):
            var value: Float64 = 0
            for j in range(n):
                value += a.get(i*n+j)*b.get(j)
            dst.put(i,value)
    else:
        # Partial pivoting, scaled singularity threshold. Fixed native storage.
        var work = SIMD[.float64, 16](0)
        var inverse = SIMD[.float64, 16](0)
        var scale: Float64 = 0
        for i in range(n*n):
            work[i] = a.get(i)
            scale = max(scale,abs(work[i]))
        for i in range(n):
            inverse[i*n+i] = 1
        var det: Float64 = 1
        for k in range(n):
            var pivot = k
            for i in range(k+1,n):
                if abs(work[i*n+k]) > abs(work[pivot*n+k]):
                    pivot = i
            if abs(work[pivot*n+k]) <= 1e-14*scale:
                if op == 3:
                    return PythonObject(0.0)
                raise Error("singular or numerically singular matrix")
            if pivot != k:
                det = -det
                for j in range(n):
                    var tmp = work[k*n+j]
                    work[k*n+j] = work[pivot*n+j]
                    work[pivot*n+j] = tmp
                    tmp = inverse[k*n+j]
                    inverse[k*n+j] = inverse[pivot*n+j]
                    inverse[pivot*n+j] = tmp
            var diagonal = work[k*n+k]
            det *= diagonal
            for j in range(n):
                work[k*n+j] /= diagonal
                inverse[k*n+j] /= diagonal
            for i in range(n):
                if i != k:
                    var multiplier = work[i*n+k]
                    for j in range(n):
                        work[i*n+j] -= multiplier*work[k*n+j]
                        inverse[i*n+j] -= multiplier*inverse[k*n+j]
        if op == 3:
            return PythonObject(det)
        for i in range(n*n):
            dst.put(i,inverse[i])
    return out_obj
