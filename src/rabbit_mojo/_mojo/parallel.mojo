"""Linux POSIX threads; every worker executes only native Mojo, never CPython."""
from std.ffi import external_call
from std.python import PythonObject
from std.collections import Array
from views import View

@fieldwise_init
struct Task(Copyable, Movable):
    var source: Int
    var destination: Int
    var row_stride: Int
    var column_stride: Int
    var start: Int
    var stop: Int
    var matrix: Array[Float64,12]

@export
def rabbit_transform_worker(argument: Int) abi("C") -> Int:
    var task = Pointer[Task,ImmUntrackedOrigin](unsafe_from_address=argument)
    var source = Pointer[Float64,ImmUntrackedOrigin](unsafe_from_address=task[].source)
    var destination = Pointer[Float64,MutUntrackedOrigin](unsafe_from_address=task[].destination)
    for i in range(task[].start,task[].stop):
        var offset = i*task[].row_stride
        var x = source[unsafe_offset=offset]
        var y = source[unsafe_offset=offset+task[].column_stride]
        var z = source[unsafe_offset=offset+2*task[].column_stride]
        for j in range(3):
            destination[unsafe_offset=i*3+j] = task[].matrix[j*4]*x+task[].matrix[j*4+1]*y+task[].matrix[j*4+2]*z+task[].matrix[j*4+3]
    return 0

def parallel_transform(points_obj: PythonObject, matrix_obj: PythonObject, out_obj: PythonObject, workers_obj: PythonObject) raises -> PythonObject:
    var points = View[.float64](points_obj)
    var matrix = View[.float64](matrix_obj)
    var dst = View[.float64](out_obj)
    var workers = Int(py=workers_obj)
    var n = points.shape[0]
    var tasks = List[Task]()
    var threads = List[UInt64]()
    var coefficients = Array[Float64,12](fill=0)
    for j in range(12):
        coefficients[j] = matrix.get(j)
    # Finish allocation before publishing task addresses to worker threads.
    for i in range(workers):
        tasks.append(Task(Int(points.data),Int(dst.data),points.strides[0],points.strides[1],n*i//workers,n*(i+1)//workers,coefficients.copy()))
        threads.append(0)
    var created = 0
    var failed = False
    for i in range(workers):
        var status = external_call["pthread_create",Int32](Pointer(to=threads[i]),Int(0),rabbit_transform_worker,Int(Pointer(to=tasks[i])))
        if status != 0:
            failed = True
            break
        created += 1
    for i in range(created):
        # Valid handles, each joined exactly once, caller is never a worker.
        _ = external_call["pthread_join",Int32](threads[i],Int(0))
    # Keep owners visibly alive across all native workers and joins.
    _ = tasks
    _ = points.owner
    _ = dst.owner
    if failed:
        raise Error("pthread_create failed; all started workers were joined")
    return out_obj
