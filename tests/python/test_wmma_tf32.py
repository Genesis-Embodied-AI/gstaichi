"""Test WMMA TF32 Tensor Core operations (sm_80+ / Ampere and newer)."""

import numpy as np
import pytest

import gstaichi as ti
from gstaichi.lang.simt import warp
from gstaichi.lang import snode


def test_wmma_intrinsic_registration():
    """Test that WMMA intrinsics are properly registered in the Python API."""
    # These should not throw - just check the functions exist
    assert hasattr(warp, '_wmma_load_a_tf32')
    assert hasattr(warp, '_wmma_load_b_tf32')
    assert hasattr(warp, '_wmma_load_c_f32')
    assert hasattr(warp, '_wmma_mma_tf32')
    assert hasattr(warp, '_wmma_store_d_f32')


def test_wmma_tf32_basic():
    """Basic WMMA TF32 test with a single 16x16 tile using random matrices.

    This test requires:
    - CUDA GPU with compute capability 8.0+ (Ampere)
    - GsTaichi built with CUDA support
    """
    ti.init(arch=ti.cuda)

    # Check compute capability - skip if not sm_80+
    if ti.lang.impl.get_cuda_compute_capability() < 80:
        pytest.skip("WMMA TF32 requires sm_80+")

    # m16n16k8 configuration - one WMMA tile
    M, N, K = 16, 16, 8

    # Generate random input matrices with NumPy (uniform in [-1, 1])
    rng = np.random.default_rng(seed=12345)
    A_np = rng.uniform(-1, 1, (M, K)).astype(np.float32)
    B_np = rng.uniform(-1, 1, (K, N)).astype(np.float32)
    C_init_np = rng.uniform(-1, 1, (M, N)).astype(np.float32)

    # Compute ground truth with NumPy: D = A @ B + C
    expected = A_np @ B_np + C_init_np

    # Use fields so we can get their addresses
    # Note: B is stored transposed as (N, K) to match WMMA's col-major expectation
    A = ti.field(dtype=ti.f32, shape=(M, K))
    B_T = ti.field(dtype=ti.f32, shape=(N, K))  # B transposed
    C = ti.field(dtype=ti.f32, shape=(M, N))

    # Copy data to Taichi fields
    A.from_numpy(A_np)
    B_T.from_numpy(B_np.T)  # Transpose B for WMMA column-major layout
    C.from_numpy(C_init_np)

    @ti.kernel
    def wmma_matmul():
        # Launch 32 threads (one warp) - all must execute WMMA together
        # WMMA is warp-cooperative: all 32 threads in the warp participate
        ti.loop_config(block_dim=32)
        for _ in range(32):  # 32 work items = 1 block of 32 threads
            # Get base pointers using snode.get_addr
            a_ptr = snode.get_addr(A, [0, 0])
            b_ptr = snode.get_addr(B_T, [0, 0])
            c_ptr = snode.get_addr(C, [0, 0])

            # Load fragments - distributed across warp threads
            # B_T is stored as (N×K) row-major = K×N column-major
            a_frag = warp._wmma_load_a_tf32(a_ptr, K)  # A is M×K row-major, stride=K
            b_frag = warp._wmma_load_b_tf32(b_ptr, K)  # B as K×N col-major, stride=K
            c_frag = warp._wmma_load_c_f32(c_ptr, N)   # C is M×N row-major, stride=N

            # Matrix multiply-accumulate: D = A @ B + C
            d_frag = warp._wmma_mma_tf32(a_frag, b_frag, c_frag)

            # Store result
            warp._wmma_store_d_f32(c_ptr, d_frag, N)

    wmma_matmul()

    # TF32 has 10-bit mantissa (~0.1% relative precision)
    np.testing.assert_allclose(C.to_numpy(), expected, rtol=1e-2, atol=1e-3)


def test_wmma_tf32_large():
    """WMMA TF32 test with large non-aligned matrices (22x312 @ 312x47)."""
    ti.init(arch=ti.cuda)

    if ti.lang.impl.get_cuda_compute_capability() < 80:
        pytest.skip("WMMA TF32 requires sm_80+")

    # Non-aligned dimensions
    M, K, N = 22, 312, 47

    # WMMA tile sizes for m16n16k8
    TILE_M, TILE_N, TILE_K = 16, 16, 8

    # Pad to tile boundaries
    M_padded = ((M + TILE_M - 1) // TILE_M) * TILE_M  # 32
    K_padded = ((K + TILE_K - 1) // TILE_K) * TILE_K  # 312
    N_padded = ((N + TILE_N - 1) // TILE_N) * TILE_N  # 48

    # Number of tiles
    num_tiles_m = M_padded // TILE_M
    num_tiles_n = N_padded // TILE_N
    num_tiles_k = K_padded // TILE_K

    # Generate random matrices and pad with zeros
    rng = np.random.default_rng(seed=54321)
    A_np = np.zeros((M_padded, K_padded), dtype=np.float32)
    B_np = np.zeros((K_padded, N_padded), dtype=np.float32)
    C_np = np.zeros((M_padded, N_padded), dtype=np.float32)

    A_np[:M, :K] = rng.uniform(-1, 1, (M, K)).astype(np.float32)
    B_np[:K, :N] = rng.uniform(-1, 1, (K, N)).astype(np.float32)

    # Ground truth (only valid region)
    expected = (A_np[:M, :K] @ B_np[:K, :N]).astype(np.float32)

    # Taichi fields with padded sizes
    A = ti.field(dtype=ti.f32, shape=(M_padded, K_padded))
    B_T = ti.field(dtype=ti.f32, shape=(N_padded, K_padded))  # Transposed
    C = ti.field(dtype=ti.f32, shape=(M_padded, N_padded))

    A.from_numpy(A_np)
    B_T.from_numpy(B_np.T)
    C.from_numpy(C_np)

    @ti.kernel
    def wmma_matmul_tiled():
        # Each warp handles one (tile_m, tile_n) output tile
        # Loop over all output tiles in parallel
        ti.loop_config(block_dim=32)
        for tile_idx in range(num_tiles_m * num_tiles_n * 32):
            tile_m = (tile_idx // 32) // num_tiles_n
            tile_n = (tile_idx // 32) % num_tiles_n

            # Base offsets for this tile
            m_off = tile_m * TILE_M
            n_off = tile_n * TILE_N
            c_ptr = snode.get_addr(C, [m_off, n_off])

            # Accumulate over K tiles
            # TODO: register accumulation requires tensor local variable support
            for tile_k in range(num_tiles_k):
                k_off = tile_k * TILE_K

                a_ptr = snode.get_addr(A, [m_off, k_off])
                b_ptr = snode.get_addr(B_T, [n_off, k_off])

                a_frag = warp._wmma_load_a_tf32(a_ptr, K_padded)
                b_frag = warp._wmma_load_b_tf32(b_ptr, K_padded)
                c_frag = warp._wmma_load_c_f32(c_ptr, N_padded)

                d_frag = warp._wmma_mma_tf32(a_frag, b_frag, c_frag)
                warp._wmma_store_d_f32(c_ptr, d_frag, N_padded)

    wmma_matmul_tiled()

    # Extract valid region and compare
    # Relaxed tolerance due to TF32 precision loss accumulated over many K-tiles
    C_result = C.to_numpy()[:M, :N]
    np.testing.assert_allclose(C_result, expected, rtol=1e-2, atol=5e-3)
