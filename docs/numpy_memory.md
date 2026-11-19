# NumPy memory and ownership

`bindings.array` is the public safety gate. It accepts existing native-endian NumPy arrays, checks element alignment, rank, dtype and shape, walks the owner/base chain, and verifies `byte_bounds` against the backing allocation. Unknown owners whose bounds cannot be established fail explicitly. `as_strided` views outside the allocation fail; an address alone is not proof of safety. Writable arrays additionally need the NumPy writable flag and a conservative non-overlap proof using sorted absolute strides. Some unusual non-overlapping layouts may be conservatively rejected for writing.

No numerical input is implicitly coerced or made contiguous. C-contiguous scale calls at ranks 1/2/3 use Mojo's documented [`from_numpy_tensor`](https://mojolang.org/docs/std/python/numpy/from_numpy_tensor/) / `NumPyView[dtype,rank]`. It borrows a `Span` and shape. Official NumPyView does not accept arbitrary strides. Other calls use the small `View[dtype]` in `_mojo/views.mojo`: it retains a strong PythonObject owner and extracts data address, native shape and signed element strides once at the boundary. Its logical row-major index is resolved to physical storage in native Mojo. Negative strides start at the view's actual data address, never the base array's address. This is an explicit pointer/shape/stride view, not an invented generic NumPyView API.

| Object | Ownership and copying |
|---|---|
| ndarray input | Python owns allocation; native view borrows bytes |
| PythonObject in View | Reference-counted Python wrapper retains allocation/base chain |
| View shape/stride Lists | Small native metadata allocations/copies; array bytes are not copied |
| NumPyView Span | Scoped borrowed contiguous elements with origin tied to local PythonObject |
| caller-provided output | Python owns allocation; Mojo writes bytes in place |
| newly returned ndarray | Python wrapper explicitly allocates output; Mojo fills it |
| Vec[dtype,N] | Copies 2/3/4 elements into an owned native fixed Array |
| QR solver | Borrows source, copies values into native owned work Lists |
| dataclass Material | Copies scalar configuration fields into native struct |

Storage-sharing evidence is observable: examples/tests preserve `ctypes.data`, assert returned `is` identity for in-place scale, modify offset/reverse/Fortran views, and verify writes in the Python allocation. Caller-output tests verify returned object identity and native values. These show no array-data copy for those paths; they do not claim zero allocations across the boundary. Metadata, Python wrappers, validation temporaries and result arrays can allocate. Benchmarks separate explicit f32→f64 copying from end-to-end kernel calls.

Float64 is used by general kernels. Float32 and int64 borrowing is demonstrated by scaling; generic vectors also support float32. Integer arithmetic is validated for factor/product overflow before execution, and the factor stays integer across the native boundary (including integers above 2^53). Unsupported dtypes, byte orders, fractional element strides and misaligned addresses fail. SIMD loads require only element alignment (8 bytes), not 32-byte alignment, and tails use scalar code. Empty arrays never dereference their pointers. Read-only arrays are valid inputs to nonmutating operations and invalid outputs.

`output` rejects overlap with any input. Sequential transforms also support exactly `out is points`: x/y/z are loaded before any result from that row is stored. Partially overlapping views fail. Writable layouts cannot overlap themselves. Tensor rotations always require separate output to avoid clobbering source entries.

Pointers never leave synchronous calls or enter public long-lived native objects. Transform3D retains a Python-owned matrix; each call creates a scoped View. Counter stores a scalar only. Parallel tasks retain no Python objects and operate on disjoint output rows. The caller retains the source/output owners and the native task List through every `pthread_join`. A last-use lifetime regression was caught during development; explicit post-join uses now keep all required owners/task storage alive. Worker functions contain no Python reference-counting or API calls.

Thread safety assumes callers do not mutate input allocations concurrently from external native threads. The calling thread holds the GIL throughout; the worker threads need no GIL because they touch only native memory. There is no asynchronous borrowing API.

Verified cases are in `tests/test_numpy_views.py` and geometry/interpolation tests: 1D/2D/3D C arrays, dtype variants, caller outputs, nonzero offsets, nonunit/negative strides, Fortran order, non-contiguous 2D, read-only, empty, misaligned, overlapping and out-of-allocation layouts.
