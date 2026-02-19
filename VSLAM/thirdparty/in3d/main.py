import moderngl
import moderngl_window as mglw
import numpy as np

window_cls = mglw.get_local_window_cls("glfw")
window = window_cls(size=(512, 512), title="ModernGL Window", gl_version=(3, 3))
ctx = window.ctx
mglw.activate_context(window, ctx=ctx)
prog = ctx.program(
    vertex_shader="""
        #version 330

        in vec2 in_vert;
        in vec3 in_color;

        out vec3 v_color;

        void main() {
            v_color = in_color;
            gl_Position = vec4(in_vert, 0.0, 1.0);
        }
    """,
    fragment_shader="""
        #version 330

        in vec3 v_color;

        out vec3 f_color;

        void main() {
            f_color = v_color;
        }
    """,
)

vertices = np.asarray(
    [-0.75, -0.75, 1, 0, 0, 0.75, -0.75, 0, 1, 0, 0.0, 0.649, 0, 0, 1], dtype="f4"
)

vbo = ctx.buffer(vertices.tobytes())
vao = ctx.vertex_array(prog, vbo, "in_vert", "in_color")

while not window.is_closing:
    window.use()
    window.clear(0.0, 0.0, 0.0, 1.0)
    vao.render()
    window.swap_buffers()
