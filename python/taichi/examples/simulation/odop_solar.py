# type: ignore

import math
import pygame
import numpy as np
import taichi as ti

ti.init()


@ti.data_oriented
class SolarSystem:
    def __init__(self, n, dt):  # Initializer of the solar system simulator
        self.n = n
        self.dt = dt
        self.x = ti.Vector.field(2, dtype=ti.f32, shape=n)
        self.v = ti.Vector.field(2, dtype=ti.f32, shape=n)
        self.center = ti.Vector.field(2, dtype=ti.f32, shape=())

    @staticmethod
    @ti.func
    def random_vector(radius):  # Create a random vector in circle
        theta = ti.random() * 2 * math.pi
        r = ti.random() * radius
        return r * ti.Vector([ti.cos(theta), ti.sin(theta)])

    @ti.kernel
    def initialize_particles(self):
        # (Re)initialize particle position/velocities
        for i in range(self.n):
            offset = self.random_vector(0.5)
            self.x[i] = self.center[None] + offset  # Offset from center
            self.v[i] = [-offset.y, offset.x]  # Perpendicular to offset
            self.v[i] += self.random_vector(0.02)  # Random velocity noise
            self.v[i] *= 1 / offset.norm() ** 1.5  # Kepler's third law

    @ti.func
    def gravity(self, pos):  # Compute gravity at pos
        offset = -(pos - self.center[None])
        return offset / offset.norm() ** 3

    @ti.kernel
    def integrate(self):  # Semi-implicit Euler time integration
        for i in range(self.n):
            self.v[i] += self.dt * self.gravity(self.x[i])
            self.x[i] += self.dt * self.v[i]

    def render(self, screen, width, height):  # Render the scene on pygame screen
        # Draw sun
        sun_x = int(0.5 * width)
        sun_y = int(0.5 * height)
        pygame.draw.circle(screen, (255, 170, 136), (sun_x, sun_y), 10)  # 0xFFAA88
        
        # Draw planets
        positions = self.x.to_numpy()
        for pos in positions:
            screen_x = int(pos[0] * width)
            screen_y = int(pos[1] * height)
            if 0 <= screen_x < width and 0 <= screen_y < height:
                pygame.draw.circle(screen, (255, 255, 255), (screen_x, screen_y), 3)


def main():
    global solar

    solar = SolarSystem(8, 0.0001)
    solar.center[None] = [0.5, 0.5]
    solar.initialize_particles()

    width, height = 800, 600
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Solar System")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    solar.initialize_particles()  # reinitialize when space bar pressed.

        for _ in range(10):  # Time integration
            solar.integrate()

        # Clear screen with background color
        screen.fill((0, 113, 26))  # 0x0071A
        
        solar.render(screen, width, height)
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()


if __name__ == "__main__":
    main()
