# type: ignore

"""Physarum simulation example.

See https://sagejenson.com/physarum for the details."""

import numpy as np
import pygame
import taichi as ti

ti.init(arch=ti.gpu)

PARTICLE_N = 1024
GRID_SIZE = 512
SENSE_ANGLE = 0.20 * np.pi
SENSE_DIST = 4.0
EVAPORATION = 0.95
MOVE_ANGLE = 0.1 * np.pi
MOVE_STEP = 2.0

grid = ti.field(dtype=ti.f32, shape=[2, GRID_SIZE, GRID_SIZE])
position = ti.Vector.field(2, dtype=ti.f32, shape=[PARTICLE_N])
heading = ti.field(dtype=ti.f32, shape=[PARTICLE_N])


@ti.kernel
def init():
    for p in ti.grouped(grid):
        grid[p] = 0.0
    for i in position:
        position[i] = ti.Vector([ti.random(), ti.random()]) * GRID_SIZE
        heading[i] = ti.random() * np.pi * 2.0


@ti.func
def sense(phase, pos, ang):
    p = pos + ti.Vector([ti.cos(ang), ti.sin(ang)]) * SENSE_DIST
    return grid[phase, p.cast(int) % GRID_SIZE]


@ti.kernel
def step(phase: ti.i32):
    # move
    for i in position:
        pos, ang = position[i], heading[i]
        l = sense(phase, pos, ang - SENSE_ANGLE)
        c = sense(phase, pos, ang)
        r = sense(phase, pos, ang + SENSE_ANGLE)
        if l < c < r:
            ang += MOVE_ANGLE
        elif l > c > r:
            ang -= MOVE_ANGLE
        elif c < l and c < r:
            ang += MOVE_ANGLE * (2 * (ti.random() < 0.5) - 1)
        pos += ti.Vector([ti.cos(ang), ti.sin(ang)]) * MOVE_STEP
        position[i], heading[i] = pos, ang

    # deposit
    for i in position:
        ipos = position[i].cast(int) % GRID_SIZE
        grid[phase, ipos] += 1.0

    # diffuse
    for i, j in ti.ndrange(GRID_SIZE, GRID_SIZE):
        a = 0.0
        for di in ti.static(range(-1, 2)):
            for dj in ti.static(range(-1, 2)):
                a += grid[phase, (i + di) % GRID_SIZE, (j + dj) % GRID_SIZE]
        a *= EVAPORATION / 9.0
        grid[1 - phase, i, j] = a


def main():
    print("[Hint] Use UP/DOWN arrows to change simulation speed.")
    
    pygame.init()
    screen = pygame.display.set_mode((GRID_SIZE, GRID_SIZE))
    pygame.display.set_caption("Physarum")
    clock = pygame.time.Clock()
    
    init()
    i = 0
    step_per_frame = 1
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_UP:
                    step_per_frame = min(100, step_per_frame + 1)
                elif event.key == pygame.K_DOWN:
                    step_per_frame = max(1, step_per_frame - 1)
        
        for _ in range(step_per_frame):
            step(i % 2)
            i += 1
        
        # Convert to pygame surface
        img = grid.to_numpy()[0]
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        img_rgb = np.stack([img] * 3, axis=-1)
        surf = pygame.surfarray.make_surface(img_rgb)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()


if __name__ == "__main__":
    main()
