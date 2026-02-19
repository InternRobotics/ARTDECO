import moderngl
import abc

import numpy as np
import trimesh


class Geometry(abc.ABC):
    def __init__(self):
        self.ctx = moderngl.get_context()

    @abc.abstractmethod
    def vertex_array(self, program):
        pass

    def set_program(self, program):
        self.program = program
        self.vao = self.vertex_array(program)

    def render(self, camera, model=None, mode=None, vertices=-1):
        if model is None:
            model = np.identity(4, dtype="f4")
        self.vao.program["m_camera"].write(camera.gl_matrix())
        self.vao.program["m_model"].write(model)
        self.vao.program["m_proj"].write(camera.proj_mat.gl_matrix())
        if "viewport_size" in self.vao.program:
            self.vao.program["viewport_size"].value = camera.viewport_size
        self.vao.render(mode=mode, vertices=vertices)


class LineGeometry(Geometry):
    def __init__(self):
        super().__init__()
        self.lines = None
        self.colors = None
        self.line_buffer = None
        self.color_buffer = None
        self.vao = None
        self.program = None

    def vertex_array(self, program):
        self.line_buffer = self.ctx.buffer(self.lines)
        self.color_buffer = self.ctx.buffer(self.colors)
        return self.ctx.vertex_array(
            program,
            [
                (self.line_buffer, "4f", "in_position_w"),
                (self.color_buffer, "4f", "in_color"),
            ],
            skip_errors=True,
            mode=moderngl.LINES,
        )

    def render(self, camera, mode=None):
        if self.vao is None:
            self.vao = self.vertex_array(self.program)
        super().render(camera, mode)

    def clear(self):
        if self.vao:
            self.vao.release()
            self.vao = None
        if self.color_buffer:
            self.color_buffer.release()
            self.line_buffer.release()
            self.color_buffer = None
            self.line_buffer = None


class PointGeometry(Geometry):
    def __init__(self):
        super().__init__()
        self.points = None
        self.colors = None
        self.point_buffer = None
        self.color_buffer = None
        self.vao = None
        self.program = None

    def vertex_array(self, program):
        self.point_buffer = self.ctx.buffer(self.points)
        self.color_buffer = self.ctx.buffer(self.colors)
        return self.ctx.vertex_array(
            program,
            [
                (self.point_buffer, "3f", "in_position"),
                (self.color_buffer, "4f", "in_color"),
            ],
            skip_errors=True,
            mode=moderngl.POINTS,
        )

    def render(self, camera, mode=None):
        if self.vao is None:
            self.vao = self.vertex_array(self.program)
        super().render(camera, mode)

    def clear(self):
        if self.vao:
            self.vao.release()
            self.vao = None
        if self.color_buffer:
            self.color_buffer.release()
            self.point_buffer.release()
            self.color_buffer = None
            self.point_buffer = None


class Mesh(Geometry):
    def __init__(self, program, file_path, scale=1.0):
        super().__init__()
        mesh = trimesh.load(file_path, force="mesh")
        vertices, normals, faces = (
            scale * mesh.vertices,
            mesh.vertex_normals,
            mesh.faces,
        )

        self.vertices_gl = self.ctx.buffer(vertices.astype("f4").tobytes())
        self.normals_gl = self.ctx.buffer(normals.astype("f4").tobytes())
        self.faces_gl = self.ctx.buffer(faces.astype("i4").tobytes())

        self.vao = self.vertex_array(program)

        self.vertices = vertices
        self.normals = normals
        self.faces = faces

    def vertex_array(self, program):
        return self.ctx.vertex_array(
            program,
            [
                (self.vertices_gl, "3f", "in_position"),
                (self.normals_gl, "3f", "in_normal"),
            ],
            index_buffer=self.faces_gl,
            skip_errors=True,
            mode=moderngl.TRIANGLES,
        )


class Axis(LineGeometry):
    def __init__(self, program, length, thickness=1.0):
        super().__init__()
        self.program = program
        # fmt:off
        self.lines = np.array(
            [
                0, 0, 0, thickness, length, 0, 0, thickness,
                0, 0, 0, thickness, 0, length, 0, thickness,
                0, 0, 0, thickness, 0, 0, length, thickness,
            ], dtype=np.float32,
        )
        self.colors = np.array(
            [1, 0, 0, 1, 
             1, 0, 0, 1, 
             0, 1, 0, 1, 
             0, 1, 0, 1, 
             0, 0, 1, 1, 
             0, 0, 1, 1], dtype=np.float32,
        )
        # fmt:on


class AABB(LineGeometry):
    def __init__(self, program, aabb_min, aabb_max, thickness=1.0, color=(1, 1, 1, 1)):
        super().__init__()
        self.program = program
        min_x, min_y, min_z = aabb_min
        max_x, max_y, max_z = aabb_max
        # fmt: off
        self.lines = np.array(
            [
                min_x, min_y, min_z, thickness, min_x, min_y, max_z, thickness,
                min_x, min_y, min_z, thickness, min_x, max_y, min_z, thickness,
                min_x, min_y, max_z, thickness, min_x, max_y, max_z, thickness,
                min_x, max_y, min_z, thickness, min_x, max_y, max_z, thickness,
                max_x, min_y, min_z, thickness, max_x, min_y, max_z, thickness,
                max_x, min_y, min_z, thickness, max_x, max_y, min_z, thickness,
                max_x, min_y, max_z, thickness, max_x, max_y, max_z, thickness,
                max_x, max_y, min_z, thickness, max_x, max_y, max_z, thickness,
                min_x, min_y, min_z, thickness, max_x, min_y, min_z, thickness,
                min_x, min_y, max_z, thickness, max_x, min_y, max_z, thickness,
                min_x, max_y, min_z, thickness, max_x, max_y, min_z, thickness,
                min_x, max_y, max_z, thickness, max_x, max_y, max_z, thickness
            ], dtype=np.float32,
        )
        # fmt: on

        colors = np.array(color, dtype=np.float32)
        self.colors = np.tile(colors, (len(self.lines) // 4, 1)).reshape(-1)


class Frustum(LineGeometry):
    def __init__(
        self, program, T_CW, near, far, fov, aspect_ratio, thickness=1, color=None
    ):
        super().__init__()
        vertices, colors = self.vertices(near, far, fov, aspect_ratio, color)
        vertices = self.transform_vertices(T_CW, vertices, thickness)
        self.program = program
        self.lines = vertices
        self.colors = colors

    @staticmethod
    def transform_vertices(T_CW, vertices, thickness=1):
        v = np.dot(np.linalg.inv(T_CW), vertices.T).T.copy()
        v[:, -1] = thickness
        return v

    @staticmethod
    def vertices(
        near, far, fov, aspect_ratio, color=None, axis_length=1, axis_only=False
    ):
        if color is None:
            color = np.array([1, 1, 1, 1], dtype=np.float32)
        # use opencv coordinate
        h_near = 2 * np.tan(np.radians(fov / 2)) * near
        w_near = h_near * aspect_ratio
        h_far = 2 * np.tan(np.radians(fov / 2)) * far
        w_far = h_far * aspect_ratio

        # Near plane vertices
        near_top_left = np.array([-w_near / 2, h_near / 2, near, 1.0])
        near_top_right = np.array([w_near / 2, h_near / 2, near, 1.0])
        near_bottom_left = np.array([-w_near / 2, -h_near / 2, near, 1.0])
        near_bottom_right = np.array([w_near / 2, -h_near / 2, near, 1.0])

        # Far plane vertices
        far_top_left = np.array([-w_far / 2, h_far / 2, far, 1.0])
        far_top_right = np.array([w_far / 2, h_far / 2, far, 1.0])
        far_bottom_left = np.array([-w_far / 2, -h_far / 2, far, 1.0])
        far_bottom_right = np.array([w_far / 2, -h_far / 2, far, 1.0])
        # fmt:off
        vertices = np.array([
            near_top_left, near_top_right, near_bottom_left, near_bottom_right,
            far_top_left, far_top_right, far_bottom_left, far_bottom_right
        ])
        lines = [
            vertices[0], vertices[1], vertices[1], vertices[3], vertices[3], vertices[2], vertices[2], vertices[0],  # Near plane  
            vertices[4], vertices[5], vertices[5], vertices[7], vertices[7], vertices[6], vertices[6], vertices[4],  # Far plane
            vertices[0], vertices[4], vertices[1], vertices[5], vertices[2], vertices[6], vertices[3], vertices[7],  # Connecting lines
        ]
        axis = [np.array([0, 0, 0, 1]), np.array([axis_length, 0, 0, 1]), np.array([0, 0, 0, 1]), np.array([0, axis_length, 0, 1]), np.array([0, 0, 0, 1]), np.array([0, 0, axis_length, 1])]
        axis_color = np.array([[1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.float32)
        # fmt:on

        lines = np.array(lines, dtype=np.float32).reshape(-1, 4)
        color = np.array(color, dtype=np.float32)
        colors = np.tile(color, (len(lines), 1))
        axis = np.array(axis, dtype=np.float32).reshape(-1, 4)
        if axis_only:
            return axis, axis_color
        lines = np.concatenate([lines, axis], axis=0)
        colors = np.concatenate([colors, axis_color], axis=0)
        return lines, colors


class FrustumList(LineGeometry):
    def __init__(
        self,
        program,
        near,
        far,
        fov,
        aspect_ratio,
        thickness=1,
        color=None,
        axis_only=False,
    ):
        super().__init__()
        self.program = program
        self.lines = []
        self.colors = []
        self.near = near
        self.far = far
        self.fov = fov
        self.aspect_ratio = aspect_ratio
        self.thickness = thickness
        self.color = color

        self.frustum_vertices, self.frustum_colors = Frustum.vertices(
            self.near,
            self.far,
            self.fov,
            self.aspect_ratio,
            color=self.color,
            axis_length=0.1,
            axis_only=axis_only,
        )

    def add_pose(self, T_CW):
        vertices = Frustum.transform_vertices(
            T_CW, self.frustum_vertices, self.thickness
        )
        self.lines.append(vertices)
        self.colors.append(self.frustum_colors)

    def add_frustum(self, frustum):
        self.lines.append(frustum.lines)
        self.colors.append(frustum.colors)

    def render(self, camera, mode=None):
        if len(self.lines) == 0:
            return
        self.lines = np.concatenate(self.lines, axis=0)
        self.colors = np.concatenate(self.colors, axis=0)
        self.clear()
        super().render(camera, mode=mode)
        self.lines = []
        self.colors = []


class Pointcloud(PointGeometry):
    def __init__(self, program):
        super().__init__()
        self.program = program
        self.points = []
        self.colors = []

    def add(self, points, colors=None, uniform_color=None):
        if len(points) == 0:
            return
        points = points.reshape(-1, 3)
        self.points.append(points)
        if colors is None or uniform_color is not None:
            c = [1, 1, 1, 1] if uniform_color is None else uniform_color
            colors = np.tile(
                np.array(c, dtype=np.float32), (points.size // 3, 1)
            ).reshape(-1)
        self.colors.append(colors)

    def render(self, camera, mode=None):
        if len(self.points) == 0:
            return
        self.points = np.concatenate(self.points, axis=0)
        self.colors = np.concatenate(self.colors, axis=0)
        self.clear()
        super().render(camera, mode=mode)
        self.points = []
        self.colors = []


class DepthMesh(Geometry):
    def __init__(self, program):
        super().__init__()
        self.program = program
        self.image_tex = None
        self.depth_tex = None
        self.valid_tex = None
        self.K = None
        self.vao = self.vertex_array(program)

    def vertex_array(self, program):
        return self.ctx.vertex_array(program, [], skip_errors=True)

    def add(self, rgb, d, T_CW, K):
        h, w = d.shape
        self.K = K
        self.release()
        self.image_tex = self.ctx.texture(
            (w, h), 3, rgb.tobytes(), dtype="f4", alignment=4
        )
        self.depth_tex = self.ctx.texture(
            (w, h), 1, d.tobytes(), dtype="f4", alignment=1
        )
        v = np.ones(d.size, dtype=np.float32)
        self.valid_tex = self.ctx.texture((w, h), 1, v.tobytes(), dtype="f4")
        self.model = np.linalg.inv(T_CW).T.copy()  # transpose to convert from row->col

    def release(self):
        if self.image_tex is not None:
            self.image_tex.release()
        if self.depth_tex is not None:
            self.depth_tex.release()
        if self.valid_tex is not None:
            self.valid_tex.release()

    def render(self, camera, mode=None):
        if self.image_tex is None:
            return
        w, h = self.image_tex.size
        self.program["width"].value = w
        self.program["height"].value = h
        self.program["K"].value = (
            self.K[0, 0],
            self.K[1, 1],
            self.K[0, 2],
            self.K[1, 2],
        )

        self.image_tex.use(0)
        self.depth_tex.use(1)
        self.valid_tex.use(2)
        self.program["image"].value = 0
        self.program["depth"].value = 1
        self.program["valid"].value = 2

        super().render(camera, model=self.model, mode=moderngl.POINTS, vertices=w * h)
        self.release()


class Spheres(Geometry):
    def __init__(self, program):
        super().__init__()
        self.program = program
        self.spheres = []  # List of sphere centers
        self.radii = []  # List of sphere radii

        # Pre-cache the UV sphere's unrolled vertices
        self.cached_uv_sphere_vertices = self.create_uv_sphere(
            radius=1.0, sectors=18, stacks=18
        )

    def create_uv_sphere(self, radius=1.0, sectors=36, stacks=18):
        """
        Create a UV sphere and return an unrolled vertex array (without index buffer).
        Each triangle's vertices are directly output, and there's no need for an index buffer.
        """
        vertices = []
        pi = np.pi
        sector_step = 2 * pi / sectors
        stack_step = pi / stacks

        # Loop through stacks and sectors to generate vertices
        for i in range(stacks + 1):
            stack_angle = pi / 2 - i * stack_step  # from pi/2 to -pi/2
            xy = radius * np.cos(stack_angle)  # r * cos(u)
            z = radius * np.sin(stack_angle)  # r * sin(u)

            for j in range(sectors + 1):
                sector_angle = j * sector_step  # from 0 to 2pi

                # Vertex position (x, y, z)
                x = xy * np.cos(sector_angle)  # r * cos(u) * cos(v)
                y = xy * np.sin(sector_angle)  # r * cos(u) * sin(v)
                vertices.append([x, y, z])

        # Convert to numpy array
        vertices = np.array(vertices, dtype=np.float32)

        # Now we need to "unroll" the sphere into triangles (instead of using an index buffer)
        unrolled_vertices = []

        for i in range(stacks):
            for j in range(sectors):
                first = i * (sectors + 1) + j
                second = first + sectors + 1

                # First triangle
                unrolled_vertices.append(vertices[first])
                unrolled_vertices.append(vertices[second])
                unrolled_vertices.append(vertices[first + 1])

                # Second triangle
                unrolled_vertices.append(vertices[second])
                unrolled_vertices.append(vertices[second + 1])
                unrolled_vertices.append(vertices[first + 1])

        return np.array(unrolled_vertices, dtype=np.float32)

    def add(self, centers, radii):
        if len(centers) == 0:
            return
        centers = np.array(centers).reshape(-1, 3)
        radii = np.array(radii).reshape(-1, 1)

        self.spheres.append(centers)
        self.radii.append(radii)

    def vertex_array(self, program):
        if len(self.spheres) == 0:
            return None

        # Concatenate all spheres' centers, radii, and colors
        centers = np.concatenate(self.spheres, axis=0)
        radii = np.concatenate(self.radii, axis=0)
        # Generate vertices for all spheres
        vertices = self.generate_flattened_sphere_mesh(centers, radii)

        # Flatten the vertices and set up the vertex array
        vbo = self.ctx.buffer(vertices.astype("f4").tobytes())
        vao_content = [(vbo, "3f", "in_position")]

        # Create the VAO without an index buffer
        vao = self.ctx.vertex_array(program, vao_content)
        return vao

    def generate_flattened_sphere_mesh(self, centers, radii):
        """
        Generate non-indexed sphere vertices (unrolled version) for each sphere, applying scaling and translation.
        """
        num_spheres = centers.shape[0]
        num_vertices_per_sphere = self.cached_uv_sphere_vertices.shape[
            0
        ]  # Number of vertices per sphere

        # Repeat the cached unit sphere vertices for each sphere
        vertices = np.tile(self.cached_uv_sphere_vertices, (num_spheres, 1))

        # Scale the vertices by the respective radii (vectorized scaling)
        radii_repeated = np.repeat(radii, num_vertices_per_sphere).reshape(-1, 1)
        vertices *= radii_repeated

        # Offset by the sphere centers (vectorized translation)
        centers_repeated = np.repeat(centers, num_vertices_per_sphere, axis=0)
        vertices += centers_repeated

        return vertices

    def clear(self):
        if self.vao:
            self.vao.release()
            self.vao = None
        # Clear data after rendering
        self.spheres = []
        self.radii = []

    def render(self, camera, model=None, mode=None):
        if len(self.spheres) == 0:
            return

        self.set_program(self.program)
        super().render(camera, model, mode)
        self.clear()
