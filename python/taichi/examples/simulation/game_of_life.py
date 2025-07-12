# type: ignore

# Game of Life written in 100 lines of Taichi
# In memory of John Horton Conway (1937 - 2020)

import numpy as np
import pygame

import taichi as ti

ti.init()

n = 64
cell_size = 8
img_size = n * cell_size
alive = ti.field(int, shape=(n, n))  # alive = 1, dead = 0
count = ti.field(int, shape=(n, n))  # count of neighbours


@ti.func
def get_alive(i, j):
    return alive[i, j] if 0 <= i < n and 0 <= j < n else 0


@ti.func
def get_count(i, j):
    return (
        get_alive(i - 1, j)
        + get_alive(i + 1, j)
        + get_alive(i, j - 1)
        + get_alive(i, j + 1)
        + get_alive(i - 1, j - 1)
        + get_alive(i + 1, j - 1)
        + get_alive(i - 1, j + 1)
        + get_alive(i + 1, j + 1)
    )


# See https://www.conwaylife.com/wiki/Cellular_automaton#Rules for more rules
B, S = [3], [2, 3]
# B, S = [2], [0]


@ti.func
def calc_rule(a, c):
    if a == 0:
        for t in ti.static(B):
            if c == t:
                a = 1
    elif a == 1:
        a = 0
        for t in ti.static(S):
            if c == t:
                a = 1
    return a


@ti.kernel
def run():
    for i, j in alive:
        count[i, j] = get_count(i, j)

    for i, j in alive:
        alive[i, j] = calc_rule(alive[i, j], count[i, j])


@ti.kernel
def init():
    for i, j in alive:
        if ti.random() > 0.8:
            alive[i, j] = 1
        else:
            alive[i, j] = 0


def main():
    pygame.init()
    screen = pygame.display.set_mode((img_size, img_size))
    pygame.display.set_caption("Game of Life")
    clock = pygame.time.Clock()

    print("[Hint] Press `r` to reset")
    print("[Hint] Press SPACE to pause")
    print("[Hint] Click LMB, RMB and drag to add alive / dead cells")

    init()
    paused = False
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    alive.fill(0)

        # Handle mouse input
        if pygame.mouse.get_pressed()[0] or pygame.mouse.get_pressed()[2]:  # LMB or RMB
            mx, my = pygame.mouse.get_pos()
            grid_x, grid_y = int(mx * n / img_size), int(my * n / img_size)
            if 0 <= grid_x < n and 0 <= grid_y < n:
                alive[grid_x, grid_y] = pygame.mouse.get_pressed()[0]  # LMB = alive, RMB = dead
                paused = True

        if not paused:
            run()

        # Convert to pygame surface
        img = alive.to_numpy()
        # Resize using numpy kron for visualization
        img = np.kron(img, np.ones((cell_size, cell_size), dtype=np.uint8)) * 255
        img_rgb = np.stack([img] * 3, axis=-1)
        surf = pygame.surfarray.make_surface(img_rgb)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(15)

    pygame.quit()


if __name__ == "__main__":
    main()
