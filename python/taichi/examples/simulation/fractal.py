# type: ignore

import numpy as np
import pygame

import taichi as ti

ti.init(arch=ti.gpu)


@ti.func
def complex_sqr(z: ti.template()) -> None:
    return ti.Vector([z[0] ** 2 - z[1] ** 2, z[1] * z[0] * 2])


@ti.kernel
def paint(n: int, t: float, pixels: ti.Template) -> None:
    for i, j in pixels:  # Parallelized over all pixels
        c = ti.Vector([-0.8, ti.cos(t) * 0.2])
        z = ti.Vector([i / n - 1, j / n - 0.5]) * 2
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = complex_sqr(z) + c
            iterations += 1
        pixels[i, j] = 1 - iterations * 0.02


def main():
    n = 320
    pixels = ti.field(dtype=float, shape=(n * 2, n))
    pygame.init()
    screen = pygame.display.set_mode((n * 2, n))
    pygame.display.set_caption("Julia Set")
    clock = pygame.time.Clock()
    t = 0.0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        paint(n, t, pixels)
        t += 0.03
        img = pixels.to_numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        img_rgb = np.stack([img] * 3, axis=-1)
        surf = pygame.surfarray.make_surface(img_rgb)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    main()
