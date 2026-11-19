from std.python import Python, PythonObject

@fieldwise_init
struct Material(Copyable, Movable):
    var young: Float64
    var poisson: Float64

    def lame(self) -> Float64:
        return self.young * self.poisson / ((1 + self.poisson) * (1 - 2 * self.poisson))

    def shear(self) -> Float64:
        return self.young / (2 * (1 + self.poisson))

@fieldwise_init
struct Counter(Copyable, Movable, Writable):
    var count: Int

    @staticmethod
    def py_init(out self: Counter, args: PythonObject, kwargs: PythonObject) raises:
        if len(args) != 1 or kwargs._obj_ptr:
            raise Error("Counter expects one positional integer")
        self = Self(Int(py=args[0]))

    @staticmethod
    def increment(py_self: PythonObject) raises -> PythonObject:
        var ptr = py_self.downcast_value_ptr[Self]()
        if ptr[].count == 9223372036854775807:
            raise Error("Counter overflow")
        ptr[].count += 1
        return PythonObject(ptr[].count)

    @staticmethod
    def zero() raises -> PythonObject:
        return PythonObject(alloc=Self(0))

    @staticmethod
    def value(py_self: PythonObject) raises -> PythonObject:
        return PythonObject(py_self.downcast_value_ptr[Self]()[].count)

def scalars(values: PythonObject) raises -> PythonObject:
    var integer = Int(py=values[0])
    var fixed = Int32(integer)
    var real = Float64(py=values[1])
    var narrow = Float32(real)
    var flag = Bool(values[2])
    var text = String(py=values[3])
    var optional = Optional[Float64]()
    if not Bool(values[4] is None):
        optional = Float64(py=values[4])
    var result_optional = Python.none()
    if case .Some(x) = optional:
        result_optional = PythonObject(x * 2)
    return Python.tuple(integer, fixed, real, narrow, not flag, text + " 🐇", result_optional)

def collections(values: PythonObject) raises -> PythonObject:
    var items = List[Float64]()
    for item in values[0]:
        items.append(Float64(py=item))
    var coordinate = (Float64(py=values[1][0]), Float64(py=values[1][1]))
    var settings = Dict[String, Float64]()
    settings["scale"] = Float64(py=values[2]["scale"])
    var mode = Int(py=values[3])
    var total: Float64 = 0
    for item in items:
        total += item
    var copied = Python.list()
    for item in items:
        _ = copied.append(item)
    var record = (mode, String("native record"))
    var converted = Python.tuple(copied, Python.dict(scale=settings["scale"]), Python.tuple(record[0],record[1]))
    return Python.tuple(total * settings["scale"], coordinate[0] + coordinate[1], mode, values[2], converted)

def material(config: PythonObject) raises -> PythonObject:
    var m = Material(Float64(py=config.youngs_modulus), Float64(py=config.poissons_ratio))
    return Python.tuple(m.lame(), m.shear())

def fail(message: PythonObject) raises -> PythonObject:
    raise Error(String(py=message))

def native_affine(value: PythonObject, var **kwargs: PythonObject) raises -> PythonObject:
    for key in kwargs:
        var name = key
        if name != "factor" and name != "offset":
            raise Error("expected only factor and offset keywords")
    var factor: Float64 = 2
    var offset: Float64 = 0
    if "factor" in kwargs:
        factor = Float64(py=kwargs["factor"])
    if "offset" in kwargs:
        offset = Float64(py=kwargs["offset"])
    return PythonObject(Float64(py=value)*factor+offset)

def stiffness(config: PythonObject, out_obj: PythonObject) raises -> PythonObject:
    from views import View
    var m = Material(Float64(py=config.youngs_modulus),Float64(py=config.poissons_ratio))
    var dst = View[.float64](out_obj)
    for i in range(36):
        dst.put(i,0)
    for i in range(3):
        for j in range(3):
            dst.put(i*6+j,m.lame())
        dst.put(i*6+i,m.lame()+2*m.shear())
        dst.put((i+3)*6+i+3,m.shear())
    return out_obj

def material_roundtrip(config: PythonObject) raises -> PythonObject:
    var m = Material(Float64(py=config.youngs_modulus),Float64(py=config.poissons_ratio))
    return Python.tuple(m.young,m.poisson)
