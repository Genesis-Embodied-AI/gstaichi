# type: ignore

# Tutorials (Chinese):
# - https://www.bilibili.com/video/BV1UK4y177iH
# - https://www.bilibili.com/video/BV1DK411A771

import taichi as ti
import pygame
import numpy as np

ti.init(arch=ti.cpu)

spring_Y = ti.field(dtype=ti.f32, shape=())  # Young's modulus
paused = ti.field(dtype=ti.i32, shape=())
drag_damping = ti.field(dtype=ti.f32, shape=())
dashpot_damping = ti.field(dtype=ti.f32, shape=())

max_num_particles = 1024
particle_mass = 1.0
dt = 1e-3
substeps = 10

num_particles = ti.field(dtype=ti.i32, shape=())
x = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
v = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
f = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
fixed = ti.field(dtype=ti.i32, shape=max_num_particles)

# rest_length[i, j] == 0 means i and j are NOT connected
rest_length = ti.field(dtype=ti.f32, shape=(max_num_particles, max_num_particles))


@ti.kernel
def substep():
    n = num_particles[None]

    # Compute force
    for i in range(n):
        # Gravity
        f[i] = ti.Vector([0, -9.8]) * particle_mass
        for j in range(n):
            if rest_length[i, j] != 0:
                x_ij = x[i] - x[j]
                d = x_ij.normalized()

                # Spring force
                f[i] += -spring_Y[None] * (x_ij.norm() / rest_length[i, j] - 1) * d

                # Dashpot damping
                v_rel = (v[i] - v[j]).dot(d)
                f[i] += -dashpot_damping[None] * v_rel * d

    # We use a semi-implicit Euler (aka symplectic Euler) time integrator
    for i in range(n):
        if not fixed[i]:
            v[i] += dt * f[i] / particle_mass
            v[i] *= ti.exp(-dt * drag_damping[None])  # Drag damping

            x[i] += v[i] * dt
        else:
            v[i] = ti.Vector([0, 0])

        # Collide with four walls
        for d in ti.static(range(2)):
            # d = 0: treating X (horizontal) component
            # d = 1: treating Y (vertical) component

            if x[i][d] < 0:  # Bottom and left
                x[i][d] = 0  # move particle inside
                v[i][d] = 0  # stop it from moving further

            if x[i][d] > 1:  # Top and right
                x[i][d] = 1  # move particle inside
                v[i][d] = 0  # stop it from moving further


@ti.kernel
def new_particle(pos_x: ti.f32, pos_y: ti.f32, fixed_: ti.i32):
    # Taichi doesn't support using vectors as kernel arguments yet, so we pass scalars
    new_particle_id = num_particles[None]
    x[new_particle_id] = [pos_x, pos_y]
    v[new_particle_id] = [0, 0]
    fixed[new_particle_id] = fixed_
    num_particles[None] += 1

    # Connect with existing particles
    for i in range(new_particle_id):
        dist = (x[new_particle_id] - x[i]).norm()
        connection_radius = 0.15
        if dist < connection_radius:
            # Connect the new particle with particle i
            rest_length[i, new_particle_id] = 0.1
            rest_length[new_particle_id, i] = 0.1


@ti.kernel
def attract(pos_x: ti.f32, pos_y: ti.f32):
    for i in range(num_particles[None]):
        p = ti.Vector([pos_x, pos_y])
        v[i] += -dt * substeps * (x[i] - p) * 100


def main():
    width, height = 512, 512
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Explicit Mass Spring System")
    clock = pygame.time.Clock()
    
    # Initialize font for text rendering
    font = pygame.font.Font(None, 24)

    spring_Y[None] = 1000
    drag_damping[None] = 1
    dashpot_damping[None] = 100

    new_particle(0.3, 0.3, False)
    new_particle(0.3, 0.4, False)
    new_particle(0.4, 0.4, False)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused[None] = not paused[None]
                elif event.key == pygame.K_c:
                    num_particles[None] = 0
                    rest_length.fill(0)
                elif event.key == pygame.K_y:
                    if pygame.key.get_pressed()[pygame.K_LSHIFT] or pygame.key.get_pressed()[pygame.K_RSHIFT]:
                        spring_Y[None] /= 1.1
                    else:
                        spring_Y[None] *= 1.1
                elif event.key == pygame.K_d:
                    if pygame.key.get_pressed()[pygame.K_LSHIFT] or pygame.key.get_pressed()[pygame.K_RSHIFT]:
                        drag_damping[None] /= 1.1
                    else:
                        drag_damping[None] *= 1.1
                elif event.key == pygame.K_x:
                    if pygame.key.get_pressed()[pygame.K_LSHIFT] or pygame.key.get_pressed()[pygame.K_RSHIFT]:
                        dashpot_damping[None] /= 1.1
                    else:
                        dashpot_damping[None] *= 1.1
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # LMB
                    pos_x, pos_y = event.pos[0] / width, event.pos[1] / height
                    shift_pressed = pygame.key.get_pressed()[pygame.K_LSHIFT] or pygame.key.get_pressed()[pygame.K_RSHIFT]
                    new_particle(pos_x, pos_y, int(shift_pressed))

        # Handle right mouse button for attraction
        if pygame.mouse.get_pressed()[2]:  # RMB
            cursor_pos = pygame.mouse.get_pos()
            attract(cursor_pos[0] / width, cursor_pos[1] / height)

        if not paused[None]:
            for step in range(substeps):
                substep()

        X = x.to_numpy()
        n = num_particles[None]

        # Clear screen with background color
        screen.fill((221, 221, 221))  # 0xDDDDDD

        # Draw the springs
        for i in range(n):
            for j in range(i + 1, n):
                if rest_length[i, j] != 0:
                    start_pos = (int(X[i][0] * width), int(X[i][1] * height))
                    end_pos = (int(X[j][0] * width), int(X[j][1] * height))
                    pygame.draw.line(screen, (68, 68, 68), start_pos, end_pos, 2)  # 0x444444

        # Draw the particles
        for i in range(n):
            pos = (int(X[i][0] * width), int(X[i][1] * height))
            color = (255, 0, 0) if fixed[i] else (17, 17, 17)  # 0xFF0000 or 0x111111
            pygame.draw.circle(screen, color, pos, 5)

        # Draw text
        texts = [
            "Left click: add mass point (with shift to fix); Right click: attract",
            "C: clear all; Space: pause",
            f"Y: Spring Young's modulus {spring_Y[None]:.1f}",
            f"D: Drag damping {drag_damping[None]:.2f}",
            f"X: Dashpot damping {dashpot_damping[None]:.2f}"
        ]
        
        for i, text in enumerate(texts):
            text_surface = font.render(text, True, (0, 0, 0))
            screen.blit(text_surface, (10, 10 + i * 25))

        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()


if __name__ == "__main__":
    main()
