# type: ignore

# Authored by Tiantian Liu, Taichi Graphics.
import math

import pygame

import taichi as ti

ti.init(arch=ti.cpu)

# global control
paused = ti.field(ti.i32, ())

# gravitational constant 6.67408e-11, using 1 for simplicity
G = 1

# number of planets
N = 3000
# unit mass
m = 1
# galaxy size
galaxy_size = 0.4
# planet radius (for rendering)
planet_radius = 2
# init vel
init_vel = 120

# time-step size
h = 1e-4
# substepping
substepping = 10

# center of the screen
center = ti.Vector.field(2, ti.f32, ())

# pos, vel and force of the planets
# Nx2 vectors
pos = ti.Vector.field(2, ti.f32, N)
vel = ti.Vector.field(2, ti.f32, N)
force = ti.Vector.field(2, ti.f32, N)


@ti.kernel
def initialize():
    center[None] = [0.5, 0.5]
    for i in range(N):
        theta = ti.random() * 2 * math.pi
        r = (ti.sqrt(ti.random()) * 0.6 + 0.4) * galaxy_size
        offset = r * ti.Vector([ti.cos(theta), ti.sin(theta)])
        pos[i] = center[None] + offset
        vel[i] = [-offset.y, offset.x]
        vel[i] *= init_vel


@ti.kernel
def compute_force():
    # clear force
    for i in range(N):
        force[i] = [0.0, 0.0]

    # compute gravitational force
    for i in range(N):
        p = pos[i]
        for j in range(N):
            if i != j:  # double the computation for a better memory footprint and load balance
                diff = p - pos[j]
                r = diff.norm(1e-5)

                # gravitational force -(GMm / r^2) * (diff/r) for i
                f = -G * m * m * (1.0 / r) ** 3 * diff

                # assign to each particle
                force[i] += f


@ti.kernel
def update():
    dt = h / substepping
    for i in range(N):
        # symplectic euler
        vel[i] += dt * force[i] / m
        pos[i] += dt * vel[i]


def main():
    width, height = 800, 800
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("N-body problem")
    clock = pygame.time.Clock()

    initialize()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    initialize()
                elif event.key == pygame.K_SPACE:
                    paused[None] = not paused[None]

        if not paused[None]:
            for i in range(substepping):
                compute_force()
                update()

        # Clear screen
        screen.fill((0, 0, 0))

        # Draw particles
        positions = pos.to_numpy()
        for pos_particle in positions:
            # Scale positions to screen coordinates
            screen_x = int(pos_particle[0] * width)
            screen_y = int(pos_particle[1] * height)
            if 0 <= screen_x < width and 0 <= screen_y < height:
                pygame.draw.circle(screen, (255, 255, 255), (screen_x, screen_y), planet_radius)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
