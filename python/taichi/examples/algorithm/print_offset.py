# type: ignore

import taichi as ti
import pygame
import numpy as np

ti.init(arch=ti.cpu, print_ir=False)

n = 4
m = 8

a = ti.field(dtype=ti.i32)
ti.root.dense(ti.ij, (1, 2)).dense(ti.ij, 2).dense(ti.ij, 2).place(a)


@ti.kernel
def fill():
    for i, j in a:
        base = ti.get_addr(a.snode, [0, 0])
        a[i, j] = int(ti.get_addr(a.snode, [i, j]) - base) // 4


def main():
    fill()
    print(a.to_numpy())

    width, height = 256, 512
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("layout")
    clock = pygame.time.Clock()
    
    # Initialize font for text rendering
    font = pygame.font.Font(None, 30)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Clear screen with white background
        screen.fill((255, 255, 255))  # 0xFFFFFF
        
        # Draw horizontal lines
        for i in range(1, m):
            y_pos = int(i * height / m)
            pygame.draw.line(screen, (0, 0, 0), (0, y_pos), (width, y_pos), 2)  # 0x000000
        
        # Draw vertical lines
        for i in range(1, n):
            x_pos = int(i * width / n)
            pygame.draw.line(screen, (0, 0, 0), (x_pos, 0), (x_pos, height), 2)  # 0x000000
        
        # Draw text
        for i in range(n):
            for j in range(m):
                text = str(a[i, j])
                text_surface = font.render(text, True, (0, 0, 0))  # 0x0
                x_pos = int((i + 0.3) * width / n)
                y_pos = int((j + 0.75) * height / m)
                screen.blit(text_surface, (x_pos, y_pos))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()


if __name__ == "__main__":
    main()
