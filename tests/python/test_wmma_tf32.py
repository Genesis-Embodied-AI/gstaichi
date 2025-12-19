"""Test WMMA TF32 Tensor Core operations (sm_80+ / Ampere and newer)."""

import numpy as np
import pytest

import gstaichi as ti
from gstaichi.lang.simt import warp
from gstaichi.lang import snode, any_array


def test_wmma_intrinsic_registration():
    """Test that WMMA intrinsics are properly registered in the Python API."""
    assert hasattr(warp, '_wmma_load_a_tf32')
    assert hasattr(warp, '_wmma_load_b_tf32')
    assert hasattr(warp, '_wmma_load_c_f32')
    assert hasattr(warp, '_wmma_mma_tf32')
    assert hasattr(warp, '_wmma_store_d_f32')


def _skip_if_no_sm80():
    """Skip test if GPU compute capability < 8.0."""
    if ti.lang.impl.get_cuda_compute_capability() < 80:
        pytest.skip("WMMA TF32 requires sm_80+")


# WMMA tile sizes for m16n16k8
TILE_M, TILE_N, TILE_K = 16, 16, 8


def test_wmma_tf32_basic():
    """Basic WMMA TF32 test with a single 16x16 tile."""
    ti.init(arch=ti.cuda)
    _skip_if_no_sm80()

    M, N, K = 16, 16, 8

    rng = np.random.default_rng(seed=12345)
    A_np = rng.uniform(-1, 1, (M, K)).astype(np.float32)
    B_np = rng.uniform(-1, 1, (K, N)).astype(np.float32)
    C_np = rng.uniform(-1, 1, (M, N)).astype(np.float32)
    expected = A_np @ B_np + C_np

    A = ti.field(dtype=ti.f32, shape=(M, K))
    B_T = ti.field(dtype=ti.f32, shape=(N, K))
    C = ti.field(dtype=ti.f32, shape=(M, N))

    A.from_numpy(A_np)
    B_T.from_numpy(B_np.T)
    C.from_numpy(C_np)

    @ti.kernel
    def wmma_kernel():
        ti.loop_config(block_dim=32)
        for _ in range(32):
            a_frag = warp._wmma_load_a_tf32(snode.get_addr(A, [0, 0]), K)
            b_frag = warp._wmma_load_b_tf32(snode.get_addr(B_T, [0, 0]), K)
            c_frag = warp._wmma_load_c_f32(snode.get_addr(C, [0, 0]), N)
            d_frag = warp._wmma_mma_tf32(a_frag, b_frag, c_frag)
            warp._wmma_store_d_f32(snode.get_addr(C, [0, 0]), d_frag, N)

    wmma_kernel()

    np.testing.assert_allclose(C.to_numpy(), expected, rtol=1e-2, atol=1e-3)


def test_wmma_tf32_basic_ndarray():
    """Basic WMMA TF32 test with ndarray storage."""
    ti.init(arch=ti.cuda)
    _skip_if_no_sm80()

    M, N, K = 16, 16, 8

    rng = np.random.default_rng(seed=12345)
    A_np = rng.uniform(-1, 1, (M, K)).astype(np.float32)
    B_np = rng.uniform(-1, 1, (K, N)).astype(np.float32)
    C_np = rng.uniform(-1, 1, (M, N)).astype(np.float32)
    expected = A_np @ B_np + C_np

    A = ti.ndarray(dtype=ti.f32, shape=(M, K))
    B_T = ti.ndarray(dtype=ti.f32, shape=(N, K))
    C = ti.ndarray(dtype=ti.f32, shape=(M, N))

    A.from_numpy(A_np)
    B_T.from_numpy(B_np.T)
    C.from_numpy(C_np)

    @ti.kernel
    def wmma_kernel(
        A: ti.types.ndarray(dtype=ti.f32, ndim=2),
        B_T: ti.types.ndarray(dtype=ti.f32, ndim=2),
        C: ti.types.ndarray(dtype=ti.f32, ndim=2)
    ):
        ti.loop_config(block_dim=32)
        for _ in range(32):
            a_frag = warp._wmma_load_a_tf32(any_array.get_addr(A, [0, 0]), K)
            b_frag = warp._wmma_load_b_tf32(any_array.get_addr(B_T, [0, 0]), K)
            c_frag = warp._wmma_load_c_f32(any_array.get_addr(C, [0, 0]), N)
            d_frag = warp._wmma_mma_tf32(a_frag, b_frag, c_frag)
            warp._wmma_store_d_f32(any_array.get_addr(C, [0, 0]), d_frag, N)

    wmma_kernel(A, B_T, C)

    np.testing.assert_allclose(C.to_numpy(), expected, rtol=1e-2, atol=1e-3)


def test_wmma_tf32_tiled():
    """WMMA TF32 test with tiled matmul for non-aligned dimensions."""
    ti.init(arch=ti.cuda)
    _skip_if_no_sm80()

    M, K, N = 22, 47, 312

    M_padded = ((M + TILE_M - 1) // TILE_M) * TILE_M
    K_padded = ((K + TILE_K - 1) // TILE_K) * TILE_K
    N_padded = ((N + TILE_N - 1) // TILE_N) * TILE_N

    num_tiles_m = M_padded // TILE_M
    num_tiles_n = N_padded // TILE_N
    num_tiles_k = K_padded // TILE_K

    rng = np.random.default_rng(seed=54321)
    A_np = np.zeros((M_padded, K_padded), dtype=np.float32)
    B_np = np.zeros((K_padded, N_padded), dtype=np.float32)
    C_np = np.zeros((M_padded, N_padded), dtype=np.float32)

    A_np[:M, :K] = rng.uniform(-1, 1, (M, K)).astype(np.float32)
    B_np[:K, :N] = rng.uniform(-1, 1, (K, N)).astype(np.float32)
    expected = (A_np[:M, :K] @ B_np[:K, :N]).astype(np.float32)

    A = ti.field(dtype=ti.f32, shape=(M_padded, K_padded))
    B_T = ti.field(dtype=ti.f32, shape=(N_padded, K_padded))
    C = ti.field(dtype=ti.f32, shape=(M_padded, N_padded))

    A.from_numpy(A_np)
    B_T.from_numpy(B_np.T)
    C.from_numpy(C_np)

    @ti.kernel
    def wmma_tiled(
        num_m: ti.i32, num_n: ti.i32, num_k: ti.i32,
        stride_a: ti.i32, stride_b: ti.i32, stride_c: ti.i32
    ):
        ti.loop_config(block_dim=32)
        for tile_idx in range(num_m * num_n * 32):
            tile_m, tile_n = (tile_idx // 32) // num_n, (tile_idx // 32) % num_n
            m_off, n_off = tile_m * TILE_M, tile_n * TILE_N
            c_ptr = snode.get_addr(C, [m_off, n_off])

            for tile_k in range(num_k):
                k_off = tile_k * TILE_K
                a_frag = warp._wmma_load_a_tf32(snode.get_addr(A, [m_off, k_off]), stride_a)
                b_frag = warp._wmma_load_b_tf32(snode.get_addr(B_T, [n_off, k_off]), stride_b)
                c_frag = warp._wmma_load_c_f32(c_ptr, stride_c)
                d_frag = warp._wmma_mma_tf32(a_frag, b_frag, c_frag)
                warp._wmma_store_d_f32(c_ptr, d_frag, stride_c)

    wmma_tiled(num_tiles_m, num_tiles_n, num_tiles_k,
               K_padded, K_padded, N_padded)

    np.testing.assert_allclose(C.to_numpy()[:M, :N], expected, rtol=1e-2, atol=5e-3)
