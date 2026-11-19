# Recorded benchmark results

Measured end-to-end on 19 November 2026. Three warmups and nine independent timed calls per case. Values are milliseconds; spread is interquartile range. Array creation for input data is outside timings. Public API validation, result allocation and native computation are inside timings. No performance claims beyond these local measurements.

```json
{
  "python": "3.14.7",
  "numpy": "2.5.5",
  "rabbit_mojo": "2026.11.0",
  "mojo": "Mojo 1.2.0 (2016d0cd)",
  "platform": "Linux-7.0.0-43-generic-x86_64-with-glibc2.39",
  "cpu": "AMD Ryzen 7 8845HS w/ Radeon 780M Graphics",
  "logical_cpus": 16,
  "flags": "-O3 --target-cpu x86-64",
  "numpy_thread_limit": 1,
  "mojo_threads": "1 sequential/SIMD; 2 pthread workers where labelled",
  "threadpools": [
    {
      "user_api": "blas",
      "internal_api": "openblas",
      "num_threads": 1,
      "prefix": "libscipy_openblas",
      "filepath": "/home/lloydf/rabbit-mojo/.venv/lib/python3.14/site-packages/numpy.libs/libscipy_openblas64_-4b78795a.so",
      "version": "0.3.34.237.2",
      "threading_layer": "pthreads",
      "architecture": "SkylakeX"
    }
  ]
}
```

| Workload | N | Implementation | Median ms | IQR ms |
|---|---:|---|---:|---:|
| mesh_transform | 4 | numpy_vectorised | 0.001974 | 0.000200 |
| mesh_transform | 4 | mojo_scalar_public | 0.015540 | 0.001443 |
| mesh_transform | 4 | mojo_simd_public | 0.014327 | 0.001362 |
| mesh_transform | 4 | mojo_pthread_1 | 0.079122 | 0.012795 |
| mesh_transform | 4 | mojo_parallel_2 | 0.057440 | 0.026650 |
| mesh_transform | 4 | mojo_caller_output | 0.016853 | 0.000460 |
| mesh_transform | 4 | mojo_binding_and_kernel_only | 0.005401 | 0.000120 |
| mesh_transform | 4 | explicit_float32_to_float64_copy_only | 0.000311 | 0.000041 |
| mesh_transform | 4 | explicit_conversion_and_public_transform | 0.013516 | 0.000190 |
| mesh_transform | 100000 | numpy_vectorised | 2.272494 | 0.084011 |
| mesh_transform | 100000 | mojo_scalar_public | 13.854131 | 0.063752 |
| mesh_transform | 100000 | mojo_simd_public | 13.854283 | 0.069394 |
| mesh_transform | 100000 | mojo_pthread_1 | 0.557150 | 0.287491 |
| mesh_transform | 100000 | mojo_parallel_2 | 0.160779 | 0.165087 |
| mesh_transform | 100000 | mojo_caller_output | 14.684897 | 1.708571 |
| mesh_transform | 100000 | mojo_binding_and_kernel_only | 13.813193 | 0.092658 |
| mesh_transform | 100000 | explicit_float32_to_float64_copy_only | 0.032473 | 0.000070 |
| mesh_transform | 100000 | explicit_conversion_and_public_transform | 15.794417 | 1.261874 |
| tensor | 4 | numpy_all_invariants | 0.016221 | 0.002174 |
| tensor | 4 | mojo_all_invariants | 0.029056 | 0.004358 |
| tensor | 4 | numpy_eigvalsh | 0.004398 | 0.000290 |
| tensor | 4 | mojo_jacobi_principal | 0.027664 | 0.001032 |
| tensor | 10000 | numpy_all_invariants | 2.462218 | 0.021280 |
| tensor | 10000 | mojo_all_invariants | 3.168435 | 0.144338 |
| tensor | 10000 | numpy_eigvalsh | 4.395630 | 0.086016 |
| tensor | 10000 | mojo_jacobi_principal | 11.400921 | 0.100573 |
| image_sampling | 4 | numpy_vectorised_bilinear | 0.012204 | 0.000461 |
| image_sampling | 4 | mojo_bilinear | 0.020118 | 0.001815 |
| image_sampling | 50000 | numpy_vectorised_bilinear | 2.120071 | 0.022995 |
| image_sampling | 50000 | mojo_bilinear | 4.831157 | 0.058343 |

The raw JSON includes min/max and threadpool details. NumPy uses a threadpool limit of one; native parallel transforms use two freshly created POSIX threads. The scalar source may still be auto-vectorised by the compiler. Explicit SIMD here packs a point and row into four lanes, rather than four separate points. General stride resolution adds overhead. The pthread transform uses direct row/column strides and differs from the general scalar/SIMD loop; compare pthread_1 to parallel_2 to hold that kernel structure constant when inspecting thread effects. Jacobi also computes eigenvectors even when the Python caller requests only values; NumPy eigvalsh computes values only. These differences matter when interpreting timings.

The raw binding case omits Python validation; it is diagnostic and private, not a supported way to bypass checks. Conversion rows separately measure an explicit float32→float64 allocation/copy and its combined end-to-end cost. Small/large cases show that public boundary costs can dominate small workloads. Thread startup and joins are included. Neither compile-time unrolling nor SIMD is claimed to improve performance without measurements.
