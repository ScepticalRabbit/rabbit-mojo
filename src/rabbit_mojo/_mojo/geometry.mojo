from std.python import PythonObject
from views import View

def transform(points_obj: PythonObject, transform_obj: PythonObject, out_obj: PythonObject, options: PythonObject) raises -> PythonObject:
    var points = View[.float64](points_obj)
    var matrix = View[.float64](transform_obj)
    var dst = View[.float64](out_obj)
    var translate = Bool(options[0])
    var simd = Bool(options[1])
    for i in range(points.shape[0]):
        var x = points.get(i*3)
        var y = points.get(i*3+1)
        var z = points.get(i*3+2)
        if simd:
            var p = SIMD[.float64, 4](x,y,z,Float64(translate))
            for j in range(3):
                var row = SIMD[.float64, 4](matrix.get(j*4),matrix.get(j*4+1),matrix.get(j*4+2),matrix.get(j*4+3))
                dst.put(i*3+j,(p*row).reduce_add())
        else:
            for j in range(3):
                var value = matrix.get(j*4)*x + matrix.get(j*4+1)*y + matrix.get(j*4+2)*z
                if translate:
                    value += matrix.get(j*4+3)
                dst.put(i*3+j,value)
    return out_obj

def rotate_tensor(a_obj: PythonObject, r_obj: PythonObject, out_obj: PythonObject) raises -> PythonObject:
    var a = View[.float64](a_obj)
    var r = View[.float64](r_obj)
    var dst = View[.float64](out_obj)
    for batch in range(a.size//9):
        for i in range(3):
            for j in range(3):
                var value: Float64 = 0
                for k in range(3):
                    for l in range(3):
                        value += r.get(i*3+k)*a.get(batch*9+k*3+l)*r.get(j*3+l)
                dst.put(batch*9+i*3+j,value)
    return out_obj
