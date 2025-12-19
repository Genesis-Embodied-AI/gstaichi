"""Benchmark WMMA TF32 vs classical for-loop vs cuBLAS."""

import numpy as np
import time

import gstaichi as ti
from gstaichi.lang.simt import warp
from gstaichi.lang import snode
from gstaichi import linalg

# WMMA tile sizes for m16n16k8
TILE_M, TILE_N, TILE_K = 16, 16, 8
NUM_CALLS = 1000


def benchmark_wmma_tf32(M, K, N):
    """Benchmark WMMA TF32 tiled matrix multiplication."""
    ti.init(arch=ti.cuda)

    if ti.lang.impl.get_cuda_compute_capability() < 80:
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


def benchmark_cublas(M, K, N):
    """Benchmark cuBLAS via Taichi's linalg.cublas_matmul."""
    ti.init(arch=ti.cuda)

    rng = np.random.default_rng(seed=42)
    A = ti.ndarray(dtype=ti.f32, shape=(M, K))
    B = ti.ndarray(dtype=ti.f32, shape=(K, N))
    C = ti.ndarray(dtype=ti.f32, shape=(M, N))

    A.from_numpy(rng.uniform(-1, 1, (M, K)).astype(np.float32))
    B.from_numpy(rng.uniform(-1, 1, (K, N)).astype(np.float32))

    # Warmup
    for _ in range(10):
        linalg.cublas_matmul(A, B, C)
    ti.sync()

    # Benchmark: 1000 calls + 1 sync
    start = time.perf_counter()
    for _ in range(NUM_CALLS):
        linalg.cublas_matmul(A, B, C)
    ti.sync()
    end = time.perf_counter()

    return (end - start) / NUM_CALLS * 1000  # ms per call


def main():
    print("=" * 80)
    print("Matrix Multiplication Benchmark: Taichi Classical vs WMMA TF32 vs cuBLAS")
    print(f"({NUM_CALLS} kernel calls + 1 sync)")
    print("=" * 80)

    # Test different matrix sizes
    sizes = [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ]

    print(f"\n{'Size':<16} {'Classical':<12} {'WMMA TF32':<12} {'cuBLAS':<12} {'WMMA/Class':<12} {'cuBLAS/Class':<12}")
    print("-" * 76)

    for M, K, N in sizes:
        # Classical for-loop
        t_classical = benchmark_classical(M, K, N)

        # WMMA TF32
        t_wmma = benchmark_wmma_tf32(M, K, N)

        # cuBLAS
        t_cublas = benchmark_cublas(M, K, N)

        size_str = f"{M}×{K}×{N}"
        class_str = f"{t_classical:.4f}ms"
        wmma_str = f"{t_wmma:.4f}ms" if t_wmma else "N/A"
        cublas_str = f"{t_cublas:.4f}ms" if t_cublas else "N/A"

        speedup_wmma = f"{t_classical/t_wmma:.1f}x" if t_wmma else "N/A"
        speedup_cublas = f"{t_classical/t_cublas:.1f}x" if t_cublas else "N/A"

        print(f"{size_str:<16} {class_str:<12} {wmma_str:<12} {cublas_str:<12} {speedup_wmma:<12} {speedup_cublas:<12}")

    print("\n" + "=" * 80)
    print("Notes:")
    print("- WMMA TF32 requires sm_80+ (Ampere). Current impl is suboptimal (no register accum).")
    print("- cuBLAS uses NVIDIA's highly optimized library via ti.linalg.cublas_matmul().")
    print("- Classical loop is Taichi's auto-parallelized naive implementation.")
    print("=" * 80)


if __name__ == "__main__":
    main()

