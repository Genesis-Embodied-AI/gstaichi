"""Benchmark WMMA TF32 vs classical for-loop matrix multiplication."""

import numpy as np
import time

import gstaichi as ti
from gstaichi.lang.simt import warp
from gstaichi.lang import snode

# WMMA tile sizes for m16n16k8
TILE_M, TILE_N, TILE_K = 16, 16, 8
NUM_CALLS = 1000


def benchmark_wmma_tf32(M, K, N):
    """Benchmark WMMA TF32 tiled matrix multiplication."""
    ti.init(arch=ti.cuda)

    if ti.lang.impl.get_cuda_compute_capability() < 80:
        print("WMMA TF32 requires sm_80+, skipping")
        return None

    # Pad to tile boundaries
    M_padded = ((M + TILE_M - 1) // TILE_M) * TILE_M
    K_padded = ((K + TILE_K - 1) // TILE_K) * TILE_K
    N_padded = ((N + TILE_N - 1) // TILE_N) * TILE_N

    num_tiles_m = M_padded // TILE_M
    num_tiles_n = N_padded // TILE_N
    num_tiles_k = K_padded // TILE_K

    # Create fields
    A = ti.field(dtype=ti.f32, shape=(M_padded, K_padded))
    B_T = ti.field(dtype=ti.f32, shape=(N_padded, K_padded))
    C = ti.field(dtype=ti.f32, shape=(M_padded, N_padded))

    # Initialize with random data
    rng = np.random.default_rng(seed=42)
    A_np = np.zeros((M_padded, K_padded), dtype=np.float32)
    B_np = np.zeros((K_padded, N_padded), dtype=np.float32)
    A_np[:M, :K] = rng.uniform(-1, 1, (M, K)).astype(np.float32)
    B_np[:K, :N] = rng.uniform(-1, 1, (K, N)).astype(np.float32)

    A.from_numpy(A_np)
    B_T.from_numpy(B_np.T)

    @ti.kernel
    def wmma_matmul(
        num_m: ti.i32, num_n: ti.i32, num_k: ti.i32,
        stride_a: ti.i32, stride_b: ti.i32, stride_c: ti.i32
    ):
        ti.loop_config(block_dim=32)
        for tile_idx in range(num_m * num_n * 32):
            tile_m = (tile_idx // 32) // num_n
            tile_n = (tile_idx // 32) % num_n
            m_off = tile_m * TILE_M
            n_off = tile_n * TILE_N
            c_ptr = snode.get_addr(C, [m_off, n_off])

            for tile_k in range(num_k):
                k_off = tile_k * TILE_K
                a_frag = warp._wmma_load_a_tf32(snode.get_addr(A, [m_off, k_off]), stride_a)
                b_frag = warp._wmma_load_b_tf32(snode.get_addr(B_T, [n_off, k_off]), stride_b)
                c_frag = warp._wmma_load_c_f32(c_ptr, stride_c)
                d_frag = warp._wmma_mma_tf32(a_frag, b_frag, c_frag)
                warp._wmma_store_d_f32(c_ptr, d_frag, stride_c)

    # Warmup
    for _ in range(10):
        wmma_matmul(num_tiles_m, num_tiles_n, num_tiles_k, K_padded, K_padded, N_padded)
    ti.sync()

    # Benchmark: 1000 calls + 1 sync
    start = time.perf_counter()
    for _ in range(NUM_CALLS):
        wmma_matmul(num_tiles_m, num_tiles_n, num_tiles_k, K_padded, K_padded, N_padded)
    ti.sync()
    end = time.perf_counter()

    return (end - start) / NUM_CALLS * 1000  # ms per call


def benchmark_classical(M, K, N):
    """Benchmark classical for-loop matrix multiplication."""
    ti.init(arch=ti.cuda)

    A = ti.field(dtype=ti.f32, shape=(M, K))
    B = ti.field(dtype=ti.f32, shape=(K, N))
    C = ti.field(dtype=ti.f32, shape=(M, N))

    # Initialize with random data
    rng = np.random.default_rng(seed=42)
    A.from_numpy(rng.uniform(-1, 1, (M, K)).astype(np.float32))
    B.from_numpy(rng.uniform(-1, 1, (K, N)).astype(np.float32))

    @ti.kernel
    def classical_matmul():
        for i, j in ti.ndrange(M, N):
            acc = 0.0
            for k in range(K):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc

    # Warmup
    for _ in range(10):
        classical_matmul()
    ti.sync()

    # Benchmark: 1000 calls + 1 sync
    start = time.perf_counter()
    for _ in range(NUM_CALLS):
        classical_matmul()
    ti.sync()
    end = time.perf_counter()

    return (end - start) / NUM_CALLS * 1000  # ms per call


def main():
    print("=" * 70)
    print("Matrix Multiplication Benchmark: WMMA TF32 vs Classical For-Loop")
    print(f"({NUM_CALLS} kernel calls + 1 sync)")
    print("=" * 70)

    # Test different matrix sizes
    sizes = [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ]

    print(f"\n{'Size (M×K×N)':<20} {'Classical (ms)':<15} {'WMMA TF32 (ms)':<15} {'Speedup':<10}")
    print("-" * 60)

    for M, K, N in sizes:
        # Classical for-loop
        t_classical = benchmark_classical(M, K, N)

        # WMMA TF32
        t_wmma = benchmark_wmma_tf32(M, K, N)

        if t_wmma is not None:
            speedup = t_classical / t_wmma
            print(f"{M}×{K}×{N:<12} {t_classical:<15.4f} {t_wmma:<15.4f} {speedup:<10.1f}x")
        else:
            print(f"{M}×{K}×{N:<12} {t_classical:<15.4f} {'N/A':<15} {'N/A':<10}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

