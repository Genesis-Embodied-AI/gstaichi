# type: ignore

import numpy as np
import pygame

import taichi as ti

ti.init()

N = 8
dt = 1e-5

x = ti.Vector.field(2, dtype=ti.f32, shape=N, needs_grad=True)  # particle positions
v = ti.Vector.field(2, dtype=ti.f32, shape=N)  # particle velocities
U = ti.field(dtype=ti.f32, shape=(), needs_grad=True)  # potential energy


@ti.kernel
def compute_U():
    for i, j in ti.ndrange(N, N):
        r = x[i] - x[j]
        # r.norm(1e-3) is equivalent to ti.sqrt(r.norm()**2 + 1e-3)
        # This is to prevent 1/0 error which can cause wrong derivative
        U[None] += -1 / r.norm(1e-3)  # U += -1 / |r|


@ti.kernel
def advance():
    for i in x:
        v[i] += dt * -x.grad[i]  # dv/dt = -dU/dx
    for i in x:
        x[i] += dt * v[i]  # dx/dt = v


def substep():
    with ti.ad.Tape(loss=U):
        # Kernel invocations in this scope will later contribute to partial derivatives of
        # U with respect to input variables such as x.
        compute_U()  # The tape will automatically compute dU/dx and save the results in x.grad
    advance()


@ti.kernel
def init():
    for i in x:
        x[i] = [ti.random(), ti.random()]


def main():
    init()

    width, height = 800, 600
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Autodiff gravity")
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        for i in range(50):
            substep()

        # Clear screen
        screen.fill((0, 0, 0))

        # Draw particles
        positions = x.to_numpy()
        for pos in positions:
            # Scale positions to screen coordinates
            screen_x = int(pos[0] * width)
            screen_y = int(pos[1] * height)
            if 0 <= screen_x < width and 0 <= screen_y < height:
                pygame.draw.circle(screen, (255, 255, 255), (screen_x, screen_y), 3)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
