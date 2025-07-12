# type: ignore

import pygame

import taichi as ti

ti.init(arch=ti.gpu)

N = 12
dt = 5e-5
dx = 1 / N
rho = 4e1
NF = 2 * N**2  # number of faces
NV = (N + 1) ** 2  # number of vertices
E, nu = 4e4, 0.2  # Young's modulus and Poisson's ratio
mu, lam = E / 2 / (1 + nu), E * nu / (1 + nu) / (1 - 2 * nu)  # Lame parameters
ball_pos, ball_radius = ti.Vector([0.5, 0.0]), 0.31
damping = 14.5

pos = ti.Vector.field(2, float, NV, needs_grad=True)
vel = ti.Vector.field(2, float, NV)
f2v = ti.Vector.field(3, int, NF)  # ids of three vertices of each face
B = ti.Matrix.field(2, 2, float, NF)
F = ti.Matrix.field(2, 2, float, NF, needs_grad=True)
V = ti.field(float, NF)
phi = ti.field(float, NF)  # potential energy of each face (Neo-Hookean)
U = ti.field(float, (), needs_grad=True)  # total potential energy

gravity = ti.Vector.field(2, float, ())
attractor_pos = ti.Vector.field(2, float, ())
attractor_strength = ti.field(float, ())


@ti.kernel
def update_U():
    for i in range(NF):
        ia, ib, ic = f2v[i]
        a, b, c = pos[ia], pos[ib], pos[ic]
        V[i] = abs((a - c).cross(b - c))
        D_i = ti.Matrix.cols([a - c, b - c])
        F[i] = D_i @ B[i]
    for i in range(NF):
        F_i = F[i]
        log_J_i = ti.log(F_i.determinant())
        phi_i = mu / 2 * ((F_i.transpose() @ F_i).trace() - 2)
        phi_i -= mu * log_J_i
        phi_i += lam / 2 * log_J_i**2
        phi[i] = phi_i
        U[None] += V[i] * phi_i


@ti.kernel
def advance():
    for i in range(NV):
        acc = -pos.grad[i] / (rho * dx**2)
        g = gravity[None] * 0.8 + attractor_strength[None] * (attractor_pos[None] - pos[i]).normalized(1e-5)
        vel[i] += dt * (acc + g * 40)
        vel[i] *= ti.exp(-dt * damping)
    for i in range(NV):
        # ball boundary condition:
        disp = pos[i] - ball_pos
        disp2 = disp.norm_sqr()
        if disp2 <= ball_radius**2:
            NoV = vel[i].dot(disp)
            if NoV < 0:
                vel[i] -= NoV * disp / disp2
        cond = (pos[i] < 0) & (vel[i] < 0) | (pos[i] > 1) & (vel[i] > 0)
        # rect boundary condition:
        for j in ti.static(range(pos.n)):
            if cond[j]:
                vel[i][j] = 0
        pos[i] += dt * vel[i]


@ti.kernel
def init_pos():
    for i, j in ti.ndrange(N + 1, N + 1):
        k = i * (N + 1) + j
        pos[k] = ti.Vector([i, j]) / N * 0.25 + ti.Vector([0.45, 0.45])
        vel[k] = ti.Vector([0, 0])
    for i in range(NF):
        ia, ib, ic = f2v[i]
        a, b, c = pos[ia], pos[ib], pos[ic]
        B_i_inv = ti.Matrix.cols([a - c, b - c])
        B[i] = B_i_inv.inverse()


@ti.kernel
def init_mesh():
    for i, j in ti.ndrange(N, N):
        k = (i * N + j) * 2
        a = i * (N + 1) + j
        b = a + 1
        c = a + N + 2
        d = a + N + 1
        f2v[k + 0] = [a, b, c]
        f2v[k + 1] = [c, d, a]


def clip(v: int, min_v: int, max_v: int) -> int:
    return min(max_v, max(v, min_v))


def paint_phi(screen, width, height):
    pos_ = pos.to_numpy()
    phi_ = phi.to_numpy()
    f2v_ = f2v.to_numpy()
    a, b, c = pos_[f2v_[:, 0]], pos_[f2v_[:, 1]], pos_[f2v_[:, 2]]
    k = phi_ * (10 / E)
    gb = (1 - k) * 0.5

    # Draw triangles
    for i in range(len(a)):
        points = [
            (int(a[i][0] * width), height - int(a[i][1] * height)),
            (int(b[i][0] * width), height - int(b[i][1] * height)),
            (int(c[i][0] * width), height - int(c[i][1] * height)),
        ]
        color = (
            clip(int((k[i] + gb[i]) * 255), 0, 255),
            clip(int(gb[i] * 255), 0, 255),
            clip(int(gb[i] * 255), 0, 255),
        )
        pygame.draw.polygon(screen, color, points)


def main():
    init_mesh()
    init_pos()
    gravity[None] = [0, -1]

    width, height = 512, 512
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("FEM128")
    clock = pygame.time.Clock()

    print(
        "[Hint] Use WSAD/arrow keys to control gravity. Use left/right mouse buttons to attract/repel. Press R to reset."
    )

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    init_pos()
                elif event.key in (pygame.K_a, pygame.K_LEFT):
                    gravity[None] = [-1, 0]
                elif event.key in (pygame.K_d, pygame.K_RIGHT):
                    gravity[None] = [+1, 0]
                elif event.key in (pygame.K_s, pygame.K_DOWN):
                    gravity[None] = [0, -1]
                elif event.key in (pygame.K_w, pygame.K_UP):
                    gravity[None] = [0, +1]

        mouse_pos = pygame.mouse.get_pos()
        attractor_pos[None] = [mouse_pos[0] / width, mouse_pos[1] / height]
        attractor_strength[None] = pygame.mouse.get_pressed()[0] - pygame.mouse.get_pressed()[2]  # LMB - RMB

        for i in range(50):
            with ti.ad.Tape(loss=U):
                update_U()
            advance()

        # Clear screen
        screen.fill((0, 0, 0))

        # Draw triangles
        paint_phi(screen, width, height)

        # Draw mouse cursor
        pygame.draw.circle(screen, (51, 102, 153), mouse_pos, 15)  # 0x336699

        # Draw ball
        ball_screen_pos = (int(ball_pos[0] * width), int(height - ball_pos[1] * height))
        pygame.draw.circle(screen, (102, 102, 102), ball_screen_pos, int(ball_radius * width))  # 0x666666

        # Draw vertices
        positions = pos.to_numpy()
        for pos_vertex in positions:
            screen_x = int(pos_vertex[0] * width)
            screen_y = int(height - pos_vertex[1] * height)
            if 0 <= screen_x < width and 0 <= screen_y < height:
                pygame.draw.circle(screen, (255, 170, 51), (screen_x, screen_y), 2)  # 0xFFAA33

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
