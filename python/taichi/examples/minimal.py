import time

import taichi as ti


@ti.kernel
def lcg_ti(B: int, lcg_its: int, a: ti.types.NDArray[ti.i32, 1]) -> None:
    for i in range(B):
        x = a[i]
        for j in range(lcg_its):
            x = (1664525 * x + 1013904223) % 2147483647
        a[i] = x


def main() -> None:
    ti.init(arch=ti.cpu)

    B = 10
    a = ti.ndarray(ti.int32, (B,))

    ti.sync()
    start = time.time()
    lcg_ti(B, 10, a)
    ti.sync()
    end = time.time()


main()
