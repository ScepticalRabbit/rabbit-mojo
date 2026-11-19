"""Column-vector affine transforms; array storage order is independent of convention."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import numpy as np
from . import _native
from .bindings import array, output
from .types import Transform3D

def transform_points(points, transform, out=None, *, vectors=False, simd=False):
    array(points,ndim=2)
    if points.shape[1]!=3: raise ValueError("points must have shape (N,3)")
    if isinstance(transform,Transform3D): transform=transform.matrix
    array(transform,shape=(4,4))
    if not np.isfinite(transform).all() or not np.array_equal(transform[3], [0,0,0,1]):
        raise ValueError("expected finite affine transform with bottom row [0,0,0,1]")
    # Exact same view is safe: each point is read completely before writing.
    if out is points:
        array(out,shape=points.shape,writable=True)
        if np.shares_memory(out,transform): raise ValueError("output overlaps transform")
    else:
        out=output(out,points.shape,points,transform)
    return _native.transform(points,transform,out,(not vectors,simd))

def transform_vectors(vectors, transform, out=None, *, simd=False):
    return transform_points(vectors,transform,out,vectors=True,simd=simd)

def rotate_vectors(vectors,rotation,out=None):
    array(rotation,shape=(3,3))
    transform=np.eye(4); transform[:3,:3]=rotation
    return transform_vectors(vectors,transform,out)

def rotate_tensors(tensors,rotation,out=None):
    array(tensors); array(rotation,shape=(3,3))
    if tensors.shape[-2:]!=(3,3): raise ValueError("tensor shape must end in (3,3)")
    if not np.allclose(rotation@rotation.T,np.eye(3),rtol=1e-12,atol=1e-12) or not np.isclose(np.linalg.det(rotation),1):
        raise ValueError("rotation must be proper orthogonal")
    out=output(out,tensors.shape,tensors,rotation)
    return _native.rotate_tensor(tensors,rotation,out)

def _transform_chunk(args):
    return transform_points(*args)

def parallel_transform(points,transform,*,workers=2):
    """Process parallelism around native kernels; IPC copies and startup are explicit costs."""
    array(points,ndim=2); array(transform,shape=(4,4))
    if type(workers) is not int or workers<1: raise ValueError("workers must be a positive integer")
    if points.shape[1]!=3: raise ValueError("expected (N,3)")
    if not len(points): return transform_points(points,transform)
    chunks=np.array_split(points,min(workers,len(points)))
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        return np.concatenate(list(pool.map(_transform_chunk,[(chunk,transform) for chunk in chunks])))
