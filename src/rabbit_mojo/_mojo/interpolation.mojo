from std.python import PythonObject
from std.math import floor, min, max, sqrt
from views import View

def interpolate(field_obj: PythonObject, queries_obj: PythonObject, out_obj: PythonObject, options: PythonObject) raises -> PythonObject:
    var field = View[.float64](field_obj)
    var queries = View[.float64](queries_obj)
    var dst = View[.float64](out_obj)
    var nearest = Bool(options[0])
    var clamp = Bool(options[1])
    var rank = len(field.shape)
    var origin = SIMD[.float64,4](0)
    var spacing = SIMD[.float64,4](1)
    for axis in range(rank):
        origin[axis] = Float64(py=options[2][axis])
        spacing[axis] = Float64(py=options[3][axis])
    for point in range(queries.shape[0]):
        var lower = SIMD[.int64,4](0)
        var fraction = SIMD[.float64,4](0)
        for axis in range(rank):
            var x = (queries.get(point*rank+axis)-origin[axis])/spacing[axis]
            var limit = Float64(field.shape[axis]-1)
            if clamp:
                x = min(max(x,Float64(0)),limit)
            elif x < 0 or x > limit:
                raise Error("interpolation coordinate outside grid")
            if nearest:
                lower[axis] = Int64(floor(x+0.5))
            else:
                var index = min(Int(floor(x)),field.shape[axis]-2)
                lower[axis] = Int64(index)
                fraction[axis] = x-Float64(index)
        var value: Float64 = 0
        var corners = 1 << rank
        if nearest:
            corners = 1
        for corner in range(corners):
            var weight: Float64 = 1
            var offset = 0
            for axis in range(rank):
                var index = Int(lower[axis])
                if not nearest:
                    if (corner >> axis)&1:
                        index += 1
                        weight *= fraction[axis]
                    else:
                        weight *= 1-fraction[axis]
                offset = offset*field.shape[axis]+index
            value += weight*field.get(offset)
        dst.put(point,value)
    return out_obj

def shape_values(xi: Float64, eta: Float64, mut n: SIMD[.float64,4], mut dx: SIMD[.float64,4], mut dy: SIMD[.float64,4]):
    comptime for i in range(4):
        comptime sx = -1 if i == 0 or i == 3 else 1
        comptime sy = -1 if i < 2 else 1
        n[i] = (1+Float64(sx)*xi)*(1+Float64(sy)*eta)/4
        dx[i] = Float64(sx)*(1+Float64(sy)*eta)/4
        dy[i] = Float64(sy)*(1+Float64(sx)*xi)/4

def q4_shape(coords: PythonObject, n_obj: PythonObject, gradients_obj: PythonObject) raises -> PythonObject:
    var xi = Float64(py=coords[0])
    var eta = Float64(py=coords[1])
    var n = SIMD[.float64,4](0)
    var dx = SIMD[.float64,4](0)
    var dy = SIMD[.float64,4](0)
    shape_values(xi,eta,n,dx,dy)
    var dst = View[.float64](n_obj)
    var grad = View[.float64](gradients_obj)
    for i in range(4):
        dst.put(i,n[i])
        grad.put(i*2,dx[i])
        grad.put(i*2+1,dy[i])
    return n_obj

def q4_integrate(vertices_obj: PythonObject, samples_obj: PythonObject, powers_obj: PythonObject) raises -> PythonObject:
    var vertices = View[.float64](vertices_obj)
    var samples = View[.float64](samples_obj)
    var px = Int(py=powers_obj[0])
    var py = Int(py=powers_obj[1])
    var total: Float64 = 0
    for point in range(4):
        var xi: Float64 = -1/sqrt(Float64(3))
        var eta = xi
        if point&1:
            xi = -xi
        if point&2:
            eta = -eta
        var n = SIMD[.float64,4](0)
        var dx = SIMD[.float64,4](0)
        var dy = SIMD[.float64,4](0)
        shape_values(xi,eta,n,dx,dy)
        var x: Float64 = 0
        var y: Float64 = 0
        var jac = SIMD[.float64,4](0)
        for i in range(4):
            x += n[i]*vertices.get(i*2)
            y += n[i]*vertices.get(i*2+1)
            jac[0] += dx[i]*vertices.get(i*2)
            jac[1] += dy[i]*vertices.get(i*2)
            jac[2] += dx[i]*vertices.get(i*2+1)
            jac[3] += dy[i]*vertices.get(i*2+1)
        var det = jac[0]*jac[3]-jac[1]*jac[2]
        if det <= 0:
            raise Error("inverted or degenerate Q4 element")
        var value: Float64 = 1
        for _ in range(px):
            value *= x
        for _ in range(py):
            value *= y
        total += value*det
        samples.put(point*7,x)
        samples.put(point*7+1,y)
        samples.put(point*7+2,det)
        for j in range(4):
            samples.put(point*7+3+j,jac[j])
    return PythonObject(total)
