# type: ignore

# from https://github.com/taichi-dev/taichi/blob/master/docs/lang/articles/get-started/cloth_simulation.md
# migrated to pygame to work without ti.GUI (which we removed from gs-taichi)
import glfw
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_MODELVIEW,
    GL_PROJECTION,
    GL_TRIANGLES,
    glBegin,
    glClear,
    glClearColor,
    glColor3f,
    glEnable,
    glEnd,
    glLoadIdentity,
    glMatrixMode,
    glPopMatrix,
    glPushMatrix,
    glTranslatef,
    glVertex3f,
)
from OpenGL.GLU import (
    gluDeleteQuadric,
    gluLookAt,
    gluNewQuadric,
    gluPerspective,
    gluSphere,
)

import taichi as ti

ti.init(arch=ti.gpu)

n = 128
quad_size = 1.0 / n
dt = 4e-2 / n
substeps = int(1 / 60 // dt)

gravity = ti.Vector([0, -9.8, 0])
spring_Y = 3e4
dashpot_damping = 1e4
drag_damping = 1

ball_radius = 0.3
ball_center = ti.Vector.field(3, dtype=float, shape=(1,))
ball_center[0] = [0, 0, 0]

x = ti.Vector.field(3, dtype=float, shape=(n, n))
v = ti.Vector.field(3, dtype=float, shape=(n, n))

num_triangles = (n - 1) * (n - 1) * 2
indices = ti.field(int, shape=num_triangles * 3)
vertices = ti.Vector.field(3, dtype=float, shape=n * n)
colors = ti.Vector.field(3, dtype=float, shape=n * n)

bending_springs = False

spring_offsets = []


@ti.kernel
def initialize_mass_points():
    random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.1
    for i, j in x:
        x[i, j] = [i * quad_size - 0.5 + random_offset[0], 0.6, j * quad_size - 0.5 + random_offset[1]]
        v[i, j] = [0, 0, 0]


@ti.kernel
def initialize_mesh_indices():
    for i, j in ti.ndrange(n - 1, n - 1):
        quad_id = (i * (n - 1)) + j
        # 1st triangle of the square
        indices[quad_id * 6 + 0] = i * n + j
        indices[quad_id * 6 + 1] = (i + 1) * n + j
        indices[quad_id * 6 + 2] = i * n + (j + 1)
        # 2nd triangle of the square
        indices[quad_id * 6 + 3] = (i + 1) * n + j + 1
        indices[quad_id * 6 + 4] = i * n + (j + 1)
        indices[quad_id * 6 + 5] = (i + 1) * n + j
    for i, j in ti.ndrange(n, n):
        if (i // 4 + j // 4) % 2 == 0:
            colors[i * n + j] = (0.22, 0.72, 0.52)
        else:
            colors[i * n + j] = (1, 0.334, 0.52)


@ti.kernel
def substep():
    for i in ti.grouped(x):
        v[i] += gravity * dt
    for i in ti.grouped(x):
        force = ti.Vector([0.0, 0.0, 0.0])
        for spring_offset in ti.static(spring_offsets):
            j = i + spring_offset
            if 0 <= j[0] < n and 0 <= j[1] < n:
                x_ij = x[i] - x[j]
                v_ij = v[i] - v[j]
                d = x_ij.normalized()
                current_dist = x_ij.norm()
                original_dist = quad_size * float(i - j).norm()  # pylint: disable=no-member
                force += -spring_Y * d * (current_dist / original_dist - 1)
                force += -v_ij.dot(d) * d * dashpot_damping * quad_size
        v[i] += force * dt
    for i in ti.grouped(x):
        v[i] *= ti.exp(-drag_damping * dt)
        offset_to_center = x[i] - ball_center[0]
        if offset_to_center.norm() <= ball_radius:
            normal = offset_to_center.normalized()
            v[i] -= min(v[i].dot(normal), 0) * normal
        x[i] += dt * v[i]


@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(n, n):
        vertices[i * n + j] = x[i, j]


# --- PyOpenGL/GLFW rendering setup ---
def draw_cloth(vertices_np, indices_np, colors_np):
    glBegin(GL_TRIANGLES)
    for idx in range(indices_np.shape[0]):
        vi = indices_np[idx]
        color = colors_np[vi]
        glColor3f(color[0], color[1], color[2])
        vtx = vertices_np[vi]
        glVertex3f(vtx[0], vtx[1], vtx[2])
    glEnd()


def draw_ball(center, radius):
    glPushMatrix()
    glTranslatef(center[0], center[1], center[2])
    glColor3f(0.5, 0.42, 0.8)
    quadric = gluNewQuadric()
    gluSphere(quadric, radius, 32, 32)
    gluDeleteQuadric(quadric)
    glPopMatrix()


def main():
    initialize_mesh_indices()

    if bending_springs:
        for i in range(-1, 2):
            for j in range(-1, 2):
                if (i, j) != (0, 0):
                    spring_offsets.append(ti.Vector([i, j]))
    else:
        for i in range(-2, 3):
            for j in range(-2, 3):
                if (i, j) != (0, 0) and abs(i) + abs(j) <= 2:
                    spring_offsets.append(ti.Vector([i, j]))

    if not glfw.init():
        raise Exception("GLFW can't be initialized")
    window = glfw.create_window(1024, 1024, "Taichi Cloth Simulation (PyOpenGL)", None, None)
    if not window:
        glfw.terminate()
        raise Exception("GLFW window can't be created")
    glfw.make_context_current(window)
    glEnable(GL_DEPTH_TEST)
    glClearColor(1, 1, 1, 1)

    initialize_mass_points()
    current_t = 0.0

    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, 1, 0.1, 10)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0, 0, 3, 0, 0, 0, 0, 1, 0)

        if current_t > 1.5:
            initialize_mass_points()
            current_t = 0

        for _ in range(substeps):
            substep()
            current_t += dt
        update_vertices()

        # Convert Taichi fields to numpy arrays for OpenGL
        vertices_np = vertices.to_numpy()
        indices_np = indices.to_numpy()
        colors_np = colors.to_numpy()
        ball_np = ball_center.to_numpy()[0]

        draw_cloth(vertices_np, indices_np, colors_np)
        draw_ball(ball_np, ball_radius * 0.95)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    main()
