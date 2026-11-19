from std.python import PythonObject
from std.math import sqrt, abs, max
from views import View
from scalar import Material

def invariants(a_obj: PythonObject, stats_obj: PythonObject, dev_obj: PythonObject) raises -> PythonObject:
    var a = View[.float64](a_obj)
    var stats = View[.float64](stats_obj)
    var dev = View[.float64](dev_obj)
    for batch in range(a.size//9):
        var m = SIMD[.float64,16](0)
        for i in range(9):
            m[i] = a.get(batch*9+i)
        var trace = m[0]+m[4]+m[8]
        var squared: Float64 = 0
        var dev_squared: Float64 = 0
        for i in range(3):
            for j in range(3):
                squared += m[i*3+j]*m[j*3+i]
                var value = m[i*3+j]
                if i == j:
                    value -= trace/3
                dev.put(batch*9+i*3+j,value)
                dev_squared += value*value
        var det = m[0]*(m[4]*m[8]-m[5]*m[7])-m[1]*(m[3]*m[8]-m[5]*m[6])+m[2]*(m[3]*m[7]-m[4]*m[6])
        stats.put(batch*6,trace)
        stats.put(batch*6+1,(trace*trace-squared)/2)
        stats.put(batch*6+2,det)
        stats.put(batch*6+3,trace/3)
        stats.put(batch*6+4,sqrt(squared))
        stats.put(batch*6+5,dev_squared/2)
    return stats_obj

def principal(a_obj: PythonObject, values_obj: PythonObject, directions_obj: PythonObject, n_obj: PythonObject) raises -> PythonObject:
    var source = View[.float64](a_obj)
    var values = View[.float64](values_obj)
    var directions = View[.float64](directions_obj)
    var n = Int(py=n_obj)
    for batch in range(source.size//(n*n)):
        var a = SIMD[.float64,16](0)
        var v = SIMD[.float64,16](0)
        var scale: Float64 = 0
        for i in range(n*n):
            a[i] = source.get(batch*n*n+i)
            scale = max(scale,abs(a[i]))
        for i in range(n):
            v[i*n+i] = 1
        for iteration in range(100):
            var p = 0
            var q = 1
            for i in range(n):
                for j in range(i+1,n):
                    if abs(a[i*n+j]) > abs(a[p*n+q]):
                        p = i
                        q = j
            var apq = a[p*n+q]
            if abs(apq) <= 1e-15*scale:
                break
            var tau = (a[q*n+q]-a[p*n+p])/(2*apq)
            var t: Float64 = 1/(abs(tau)+sqrt(1+tau*tau))
            if tau < 0:
                t = -t
            var c = 1/sqrt(1+t*t)
            var s = t*c
            a[p*n+p] -= t*apq
            a[q*n+q] += t*apq
            a[p*n+q] = 0
            a[q*n+p] = 0
            for k in range(n):
                if k != p and k != q:
                    var akp = a[k*n+p]
                    var akq = a[k*n+q]
                    a[k*n+p] = c*akp-s*akq
                    a[p*n+k] = a[k*n+p]
                    a[k*n+q] = s*akp+c*akq
                    a[q*n+k] = a[k*n+q]
                var vkp = v[k*n+p]
                var vkq = v[k*n+q]
                v[k*n+p] = c*vkp-s*vkq
                v[k*n+q] = s*vkp+c*vkq
        # Sort eigenvalues descending, moving eigenvector columns with them.
        for i in range(n):
            var best = i
            for j in range(i+1,n):
                if a[j*n+j] > a[best*n+best]:
                    best = j
            var tmp = a[i*n+i]
            a[i*n+i] = a[best*n+best]
            a[best*n+best] = tmp
            for k in range(n):
                tmp = v[k*n+i]
                v[k*n+i] = v[k*n+best]
                v[k*n+best] = tmp
        for i in range(n):
            values.put(batch*n+i,a[i*n+i])
        for i in range(n*n):
            directions.put(batch*n*n+i,v[i])
    return values_obj

def elastic(strain_obj: PythonObject, config: PythonObject, stress_obj: PythonObject, energy_obj: PythonObject) raises -> PythonObject:
    var strain = View[.float64](strain_obj)
    var stress = View[.float64](stress_obj)
    var energy = View[.float64](energy_obj)
    var material = Material(Float64(py=config.youngs_modulus),Float64(py=config.poissons_ratio))
    var lam = material.lame()
    var mu = material.shear()
    for b in range(strain.size//9):
        var trace = strain.get(b*9)+strain.get(b*9+4)+strain.get(b*9+8)
        var value: Float64 = 0
        for i in range(9):
            var sigma = 2*mu*strain.get(b*9+i)
            if i == 0 or i == 4 or i == 8:
                sigma += lam*trace
            stress.put(b*9+i,sigma)
            value += 0.5*sigma*strain.get(b*9+i)
        energy.put(b,value)
    return stress_obj
