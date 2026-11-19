from std.python import Python, PythonObject
from std.math import abs, sqrt, cos, sin, max
from views import View

# Stencils along any logical axis. stride is in logical row-major indices;
# View resolves those indices into actual signed physical strides.
def differentiate(y_obj: PythonObject, dst_obj: PythonObject, options: PythonObject) raises -> PythonObject:
    var y = View[.float64](y_obj)
    var dst = View[.float64](dst_obj)
    var h = Float64(py=options[0])
    var axis = Int(py=options[1])
    var method = Int(py=options[2])
    var n = y.shape[axis]
    var stride = 1
    for j in range(axis+1,len(y.shape)):
        stride *= y.shape[j]
    for index in range(y.size):
        var j = (index//stride)%n
        var value: Float64
        if method == 3:
            if j == 0:
                value = (2*y.get(index)-5*y.get(index+stride)+4*y.get(index+2*stride)-y.get(index+3*stride))/(h*h)
            elif j == n-1:
                value = (2*y.get(index)-5*y.get(index-stride)+4*y.get(index-2*stride)-y.get(index-3*stride))/(h*h)
            else:
                value = (y.get(index+stride)-2*y.get(index)+y.get(index-stride))/(h*h)
        elif method == 0 and j < n-1:
            value = (y.get(index+stride)-y.get(index))/h
        elif method == 1 and j > 0:
            value = (y.get(index)-y.get(index-stride))/h
        elif j == 0:
            value = (-3*y.get(index)+4*y.get(index+stride)-y.get(index+2*stride))/(2*h)
        elif j == n-1:
            value = (3*y.get(index)-4*y.get(index-stride)+y.get(index-2*stride))/(2*h)
        else:
            value = (y.get(index+stride)-y.get(index-stride))/(2*h)
        dst.put(index,value)
    return dst_obj

def integrate(y_obj: PythonObject, x_obj: PythonObject, dst_obj: PythonObject, options: PythonObject) raises -> PythonObject:
    var y = View[.float64](y_obj)
    var x = View[.float64](x_obj)
    var dst = View[.float64](dst_obj)
    var axis = Int(py=options[0])
    var simpson = Bool(options[1])
    var n = y.shape[axis]
    var stride = 1
    for j in range(axis+1,len(y.shape)):
        stride *= y.shape[j]
    for index in range(dst.size):
        var start = (index//stride)*n*stride+index%stride
        var value: Float64 = 0
        if simpson:
            value = y.get(start)+y.get(start+(n-1)*stride)
            for i in range(1,n-1):
                var weight: Float64 = 2
                if i%2 == 1:
                    weight = 4
                value += weight*y.get(start+i*stride)
            value *= (x.get(1)-x.get(0))/3
        else:
            for i in range(n-1):
                value += (x.get(i+1)-x.get(i))*(y.get(start+i*stride)+y.get(start+(i+1)*stride))/2
        dst.put(index,value)
    return dst_obj

@fieldwise_init
struct SolveResult(Copyable, Movable):
    var root: Float64
    var converged: Bool
    var iterations: Int
    var residual: Float64

    def python(self) raises -> PythonObject:
        return Python.tuple(self.root,self.converged,self.iterations,self.residual)

def function_value(x: Float64, kind: Int) -> Float64:
    if kind == 0:
        return x*x-2
    elif kind == 1:
        return cos(x)-x
    return x*x*x-1

def derivative_value(x: Float64, kind: Int) -> Float64:
    if kind == 0:
        return 2*x
    elif kind == 1:
        return -sin(x)-1
    return 3*x*x

def newton(initial: PythonObject, options: PythonObject) raises -> PythonObject:
    var x = Float64(py=initial)
    var atol = Float64(py=options[0])
    var rtol = Float64(py=options[1])
    var iterations = Int(py=options[2])
    var kind = Int(py=options[3])
    for i in range(iterations):
        var residual = abs(function_value(x,kind))
        if residual <= atol:
            return SolveResult(x,True,i,residual).python()
        var derivative = derivative_value(x,kind)
        if abs(derivative) < 1e-30:
            raise Error("Newton derivative is zero or too small")
        var step = function_value(x,kind)/derivative
        x -= step
        if abs(step) <= atol+rtol*abs(x) and abs(function_value(x,kind)) <= atol:
            return SolveResult(x,True,i+1,abs(function_value(x,kind))).python()
    return SolveResult(x,False,iterations,abs(function_value(x,kind))).python()

# Householder QR, avoiding normal equations. Small dense educational solver.
def least_squares(a_obj: PythonObject, b_obj: PythonObject, out_obj: PythonObject) raises -> PythonObject:
    var a = View[.float64](a_obj)
    var b = View[.float64](b_obj)
    var dst = View[.float64](out_obj)
    var m = a.shape[0]
    var n = a.shape[1]
    var work = List[Float64]()
    var rhs = List[Float64]()
    var scale: Float64 = 0
    for i in range(a.size):
        work.append(a.get(i))
        scale = max(scale,abs(a.get(i)))
    for i in range(m):
        rhs.append(b.get(i))
    for k in range(n):
        var norm: Float64 = 0
        for i in range(k,m):
            norm += work[i*n+k]*work[i*n+k]
        norm = sqrt(norm)
        if norm <= 1e-12*scale*Float64(m):
            raise Error("rank deficient or ill-conditioned design matrix")
        if work[k*n+k] > 0:
            norm = -norm
        var v = List[Float64]()
        for i in range(k,m):
            v.append(work[i*n+k])
        v[0] -= norm
        var squared: Float64 = 0
        for item in v:
            squared += item*item
        for j in range(k,n):
            var product: Float64 = 0
            for i in range(k,m):
                product += v[i-k]*work[i*n+j]
            for i in range(k,m):
                work[i*n+j] -= 2*v[i-k]*product/squared
        var product: Float64 = 0
        for i in range(k,m):
            product += v[i-k]*rhs[i]
        for i in range(k,m):
            rhs[i] -= 2*v[i-k]*product/squared
    for i in range(n-1,-1,-1):
        var value = rhs[i]
        for j in range(i+1,n):
            value -= work[i*n+j]*dst.get(j)
        dst.put(i,value/work[i*n+i])
    return out_obj
