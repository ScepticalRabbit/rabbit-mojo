from `comptime` import evaluate, generic_dot
from interpolation import interpolate, q4_shape, q4_integrate
from numerical import differentiate, integrate, newton, least_squares
from tensor import invariants, principal, elastic
from geometry import transform, rotate_tensor
from linalg import dot, vector, matrix
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from scalar import scalars, collections, material, fail, Counter
from views import scale

@export
def PyInit__native() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_native")
        m.def_function[scalars]("scalars")
        m.def_function[collections]("collections")
        m.def_function[material]("material")
        m.def_function[fail]("fail")
        m.def_function[scale]("scale")
        _ = (m.add_type[Counter]("Counter")
             .def_py_init[Counter.py_init]()
             .def_method[Counter.increment]("increment")
             .def_method[Counter.value]("value"))
        m.def_function[dot]("dot")
        m.def_function[vector]("vector")
        m.def_function[matrix]("matrix")
        m.def_function[transform]("transform")
        m.def_function[rotate_tensor]("rotate_tensor")
        m.def_function[invariants]("invariants")
        m.def_function[principal]("principal")
        m.def_function[elastic]("elastic")
        m.def_function[differentiate]("differentiate")
        m.def_function[integrate]("integrate")
        m.def_function[newton]("newton")
        m.def_function[least_squares]("least_squares")
        m.def_function[interpolate]("interpolate")
        m.def_function[q4_shape]("q4_shape")
        m.def_function[q4_integrate]("q4_integrate")
        m.def_function[evaluate]("evaluate")
        m.def_function[generic_dot]("generic_dot")
        return m.finalize()
    except e:
        abort(String("rabbit_mojo initialisation failed: ", e))
