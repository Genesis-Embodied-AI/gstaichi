# type: ignore

"""GsTaichi support module for sparse matrix operations."""

from gstaichi._lib import core as _ti_core
from gstaichi.lang._ndarray import Ndarray, ScalarNdarray
from gstaichi.lang.impl import get_runtime
from gstaichi.linalg.matrixfree_cg import *
from gstaichi.linalg.sparse_cg import SparseCG
from gstaichi.linalg.sparse_matrix import *
from gstaichi.linalg.sparse_solver import SparseSolver
from gstaichi.types.primitive_types import f32

__all__ = ["SparseCG", "SparseSolver", "cublas_matmul"]


def cublas_matmul(A, B, C=None, alpha=1.0, beta=0.0, transpose_a=False, transpose_b=False):
    """Dense matrix multiplication using cuBLAS.

    Computes C = alpha * op(A) @ op(B) + beta * C

    Args:
        A: Input matrix (M x K) or (K x M) if transpose_a=True
        B: Input matrix (K x N) or (N x K) if transpose_b=True
        C: Output matrix (M x N). If None, a new ndarray is created.
        alpha: Scalar multiplier for A @ B (default: 1.0)
        beta: Scalar multiplier for C (default: 0.0)
        transpose_a: Whether to transpose A (default: False)
        transpose_b: Whether to transpose B (default: False)

    Returns:
        Ndarray: The result matrix C
    """
    if get_runtime().prog.config().arch != _ti_core.Arch.cuda:
        raise RuntimeError("cublas_matmul only supports CUDA backend")

    # Get dimensions
    if transpose_a:
        K, M = A.shape[0], A.shape[1]
    else:
        M, K = A.shape[0], A.shape[1]

    if transpose_b:
        N, K2 = B.shape[0], B.shape[1]
    else:
        K2, N = B.shape[0], B.shape[1]

    assert K == K2, f"Matrix dimension mismatch: A has {K} columns, B has {K2} rows"

    # Create output if not provided
    if C is None:
        C = ScalarNdarray(f32, [M, N])
    else:
        assert C.shape == (M, N), f"Output shape mismatch: expected ({M}, {N}), got {C.shape}"

    get_runtime().prog.cublas_sgemm(
        A.arr, B.arr, C.arr, M, N, K, alpha, beta, transpose_a, transpose_b
    )

    return C
