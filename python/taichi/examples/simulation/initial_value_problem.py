# type: ignore

import time
import pygame
import numpy as np
import taichi as ti


def init():
    a = []
    for i in np.linspace(0, 1, n, False):
        for j in np.linspace(0, 1, n, False):
            a.append([i, j])
    return np.array(a).astype(np.float32)


ti.init(arch=ti.gpu)
n = 50
dirs = ti.field(dtype=float, shape=(n * n, 2))
locations_np = init()

locations = ti.field(dtype=float, shape=(n * n, 2))
locations.from_numpy(locations_np)


@ti.kernel
def paint(t: float):
    (o, p) = locations_np.shape
    for i in range(0, o):  # Parallelized over all pixels
        x = locations[i, 0]
        y = locations[i, 1]
        dirs[i, 0] = ti.sin((t * x - y))
        dirs[i, 1] = ti.cos(t * y - x)
        l = (dirs[i, 0] ** 2 + dirs[i, 1] ** 2) ** 0.5
        dirs[i, 0] /= l * 40
        dirs[i, 1] /= l * 40


def main():
    width, height = 500, 500
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Vector Field")
    clock = pygame.time.Clock()

    beginning = time.time_ns()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        paint((time.time_ns() - beginning) * 0.00000001)
        dirs_np = dirs.to_numpy()
        
        # Clear screen
        screen.fill((0, 0, 0))
        
        # Draw arrows
        for i in range(len(locations_np)):
            start_x = int(locations_np[i, 0] * width)
            start_y = int(locations_np[i, 1] * height)
            end_x = int((locations_np[i, 0] + dirs_np[i, 0]) * width)
            end_y = int((locations_np[i, 1] + dirs_np[i, 1]) * height)
            
            if 0 <= start_x < width and 0 <= start_y < height:
                pygame.draw.line(screen, (255, 255, 255), (start_x, start_y), (end_x, end_y), 1)
                # Draw arrowhead
                if abs(end_x - start_x) > 1 or abs(end_y - start_y) > 1:
                    pygame.draw.circle(screen, (255, 255, 255), (end_x, end_y), 1)
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()


if __name__ == "__main__":
    main()
