import ctypes

import imgui
import moderngl
from imgui.integrations import compute_fb_scale
from imgui.integrations.base import BaseOpenGLRenderer


class ImguiModernglWindowMixin:
    def resize(self, width: int, height: int):
        self.io.display_size = self.wnd.size
        self.io.display_fb_scale = compute_fb_scale(self.wnd.size, self.wnd.buffer_size)
        self.io.mouse_pos = (0, 0)
        self.io.mouse_down[0] = 0
        self.io.mouse_down[1] = 0
        self.io.mouse_down[2] = 0

    def key_event(self, key, action, modifiers):
        keys = self.wnd.keys

        if action == keys.ACTION_PRESS:
            if key in self.REVERSE_KEY_MAP:
                self.io.keys_down[self.REVERSE_KEY_MAP[key]] = True
        else:
            if key in self.REVERSE_KEY_MAP:
                self.io.keys_down[self.REVERSE_KEY_MAP[key]] = False

    def _mouse_pos_viewport(self, x, y):
        """Make sure mouse coordinates are correct with black borders"""
        return (
            int(
                x
                - (self.wnd.width - self.wnd.viewport_width / self.wnd.pixel_ratio) / 2
            ),
            int(
                y
                - (self.wnd.height - self.wnd.viewport_height / self.wnd.pixel_ratio)
                / 2
            ),
        )

    def mouse_position_event(self, x, y, dx, dy):
        self.io.mouse_pos = self._mouse_pos_viewport(x, y)

    def mouse_drag_event(self, x, y, dx, dy):
        self.io.mouse_pos = self._mouse_pos_viewport(x, y)

        if self.wnd.mouse_states.left:
            self.io.mouse_down[0] = 1

        if self.wnd.mouse_states.middle:
            self.io.mouse_down[2] = 1

        if self.wnd.mouse_states.right:
            self.io.mouse_down[1] = 1

    def mouse_scroll_event(self, x_offset, y_offset):
        self.io.mouse_wheel = y_offset

    def mouse_press_event(self, x, y, button):
        self.io.mouse_pos = self._mouse_pos_viewport(x, y)

        if button == self.wnd.mouse.left:
            self.io.mouse_down[0] = 1

        if button == self.wnd.mouse.middle:
            self.io.mouse_down[2] = 1

        if button == self.wnd.mouse.right:
            self.io.mouse_down[1] = 1

    def mouse_release_event(self, x: int, y: int, button: int):
        self.io.mouse_pos = self._mouse_pos_viewport(x, y)

        if button == self.wnd.mouse.left:
            self.io.mouse_down[0] = 0

        if button == self.wnd.mouse.middle:
            self.io.mouse_down[2] = 0

        if button == self.wnd.mouse.right:
            self.io.mouse_down[1] = 0

    def unicode_char_entered(self, char):
        io = imgui.get_io()
        io.add_input_character(ord(char))


class ImguiModernGLRenderer(BaseOpenGLRenderer):
    VERTEX_SHADER_SRC = """
        #version 330
        uniform mat4 ProjMtx;
        in vec2 Position;
        in vec2 UV;
        in vec4 Color;
        out vec2 Frag_UV;
        out vec4 Frag_Color;
        void main() {
            Frag_UV = UV;
            Frag_Color = Color;
            gl_Position = ProjMtx * vec4(Position.xy, 0, 1);
        }
    """
    FRAGMENT_SHADER_SRC = """
        #version 330
        uniform sampler2D Texture;
        in vec2 Frag_UV;
        in vec4 Frag_Color;
        out vec4 Out_Color;
        void main() {
            Out_Color = (Frag_Color * texture(Texture, Frag_UV.st));
        }
    """

    def __init__(self, *args, **kwargs):
        self._prog = None
        self._fbo = None
        self._font_texture = None
        self._vertex_buffer = None
        self._index_buffer = None
        self._vao = None
        self._textures = {}
        self.wnd = kwargs.get("wnd")
        self.ctx = self.wnd.ctx if self.wnd and self.wnd.ctx else kwargs.get("ctx")

        if not self.ctx:
            raise ValueError("Missing moderngl context")

        assert isinstance(self.ctx, moderngl.Context)

        super().__init__()

        if hasattr(self, "wnd") and self.wnd:
            self.resize(*self.wnd.buffer_size)
        elif "display_size" in kwargs:
            self.io.display_size = kwargs.get("display_size")

    def register_texture(self, texture: moderngl.Texture):
        """Make the imgui renderer aware of the texture"""
        self._textures[texture.glo] = texture

    def remove_texture(self, texture: moderngl.Texture):
        """Remove the texture from the imgui renderer"""
        del self._textures[texture.glo]

    def refresh_font_texture(self):
        width, height, pixels = self.io.fonts.get_tex_data_as_rgba32()

        if self._font_texture:
            self.remove_texture(self._font_texture)
            self._font_texture.release()

        self._font_texture = self.ctx.texture((width, height), 4, data=pixels)
        self.register_texture(self._font_texture)
        self.io.fonts.texture_id = self._font_texture.glo
        self.io.fonts.clear_tex_data()

    def _create_device_objects(self):
        self._prog = self.ctx.program(
            vertex_shader=self.VERTEX_SHADER_SRC,
            fragment_shader=self.FRAGMENT_SHADER_SRC,
        )
        self.projMat = self._prog["ProjMtx"]
        self._prog["Texture"].value = 0
        self._vertex_buffer = self.ctx.buffer(reserve=imgui.VERTEX_SIZE * 65536)
        self._index_buffer = self.ctx.buffer(reserve=imgui.INDEX_SIZE * 65536)
        self._vao = self.ctx.vertex_array(
            self._prog,
            [(self._vertex_buffer, "2f 2f 4f1", "Position", "UV", "Color")],
            index_buffer=self._index_buffer,
            index_element_size=imgui.INDEX_SIZE,
        )

    def render(self, draw_data):
        io = self.io
        display_width, display_height = io.display_size
        fb_width = int(display_width * io.display_fb_scale[0])
        fb_height = int(display_height * io.display_fb_scale[1])

        if fb_width == 0 or fb_height == 0:
            return

        self.projMat.value = (
            2.0 / display_width,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0 / -display_height,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.0,
            0.0,
            -1.0,
            1.0,
            0.0,
            1.0,
        )

        draw_data.scale_clip_rects(*io.display_fb_scale)

        self.ctx.enable_only(moderngl.BLEND)
        self.ctx.blend_equation = moderngl.FUNC_ADD
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self._font_texture.use()

        for commands in draw_data.commands_lists:
            # Write the vertex and index buffer data without copying it
            vtx_type = ctypes.c_byte * commands.vtx_buffer_size * imgui.VERTEX_SIZE
            idx_type = ctypes.c_byte * commands.idx_buffer_size * imgui.INDEX_SIZE
            vtx_arr = (vtx_type).from_address(commands.vtx_buffer_data)
            idx_arr = (idx_type).from_address(commands.idx_buffer_data)
            self._vertex_buffer.write(vtx_arr)
            self._index_buffer.write(idx_arr)

            idx_pos = 0
            for command in commands.commands:
                texture = self._textures.get(command.texture_id)
                if texture is None:
                    raise ValueError(
                        (
                            "Texture {} is not registered. Please add to renderer using "
                            "register_texture(..). "
                            "Current textures: {}".format(
                                command.texture_id, list(self._textures)
                            )
                        )
                    )

                texture.use(0)

                x, y, z, w = command.clip_rect
                self.ctx.scissor = int(x), int(fb_height - w), int(z - x), int(w - y)
                self._vao.render(
                    moderngl.TRIANGLES, vertices=command.elem_count, first=idx_pos
                )
                idx_pos += command.elem_count

        self.ctx.scissor = None

    def _invalidate_device_objects(self):
        if self._font_texture:
            self._font_texture.release()
        if self._vertex_buffer:
            self._vertex_buffer.release()
        if self._index_buffer:
            self._index_buffer.release()
        if self._vao:
            self._vao.release()
        if self._prog:
            self._prog.release()

        self.io.fonts.texture_id = 0
        self._font_texture = None


class ImguiModernglWindowRenderer(ImguiModernGLRenderer, ImguiModernglWindowMixin):
    def __init__(self, window):
        super().__init__(wnd=window)
        self.wnd = window

        self._init_key_maps()
        self.io.display_size = self.wnd.size
        self.io.display_fb_scale = compute_fb_scale(self.wnd.size, self.wnd.buffer_size)
        # add itself to ctx
        self.wnd.ctx.imgui = self

    def _init_key_maps(self):
        keys = self.wnd.keys

        self.REVERSE_KEY_MAP = {
            keys.TAB: imgui.KEY_TAB,
            keys.LEFT: imgui.KEY_LEFT_ARROW,
            keys.RIGHT: imgui.KEY_RIGHT_ARROW,
            keys.UP: imgui.KEY_UP_ARROW,
            keys.DOWN: imgui.KEY_DOWN_ARROW,
            keys.PAGE_UP: imgui.KEY_PAGE_UP,
            keys.PAGE_DOWN: imgui.KEY_PAGE_DOWN,
            keys.HOME: imgui.KEY_HOME,
            keys.END: imgui.KEY_END,
            keys.DELETE: imgui.KEY_DELETE,
            keys.SPACE: imgui.KEY_SPACE,
            keys.BACKSPACE: imgui.KEY_BACKSPACE,
            keys.ENTER: imgui.KEY_ENTER,
            keys.ESCAPE: imgui.KEY_ESCAPE,
            keys.A: imgui.KEY_A,
            keys.C: imgui.KEY_C,
            keys.V: imgui.KEY_V,
            keys.X: imgui.KEY_X,
            keys.Y: imgui.KEY_Y,
            keys.Z: imgui.KEY_Z,
        }

        for value in self.REVERSE_KEY_MAP.values():
            self.io.key_map[value] = value


def docking_space(name: str):
    flags = (
        # imgui.WINDOW_MENU_BAR
        imgui.WINDOW_NO_DOCKING
        # | imgui.WINDOW_NO_BACKGROUND
        | imgui.WINDOW_NO_TITLE_BAR
        | imgui.WINDOW_NO_COLLAPSE
        | imgui.WINDOW_NO_RESIZE
        | imgui.WINDOW_NO_MOVE
        | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS
        | imgui.WINDOW_NO_NAV_FOCUS
    )

    viewport = imgui.get_main_viewport()
    x, y = viewport.pos
    w, h = viewport.size
    imgui.set_next_window_position(x, y)
    imgui.set_next_window_size(w, h)
    # imgui.set_next_window_viewport(viewport.id)
    imgui.push_style_var(imgui.STYLE_WINDOW_BORDERSIZE, 0.0)
    imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 0.0)

    # When using ImGuiDockNodeFlags_PassthruCentralNode, DockSpace() will render our background and handle the pass-thru hole, so we ask Begin() to not render a background.
    # local window_flags = self.window_flags
    # if bit.band(self.dockspace_flags, ) ~= 0 then
    #     window_flags = bit.bor(window_flags, const.ImGuiWindowFlags_.NoBackground)
    # end

    # Important: note that we proceed even if Begin() returns false (aka window is collapsed).
    # This is because we want to keep our DockSpace() active. If a DockSpace() is inactive,
    # all active windows docked into it will lose their parent and become undocked.
    # We cannot preserve the docking relationship between an active window and an inactive docking, otherwise
    # any change of dockspace/settings would lead to windows being stuck in limbo and never being visible.
    imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (0, 0))
    imgui.begin(name, None, flags)
    imgui.pop_style_var(3)

    # DockSpace
    dockspace_id = imgui.get_id(name)
    imgui.dockspace(dockspace_id, (0, 0), imgui.DOCKNODE_PASSTHRU_CENTRAL_NODE)

    imgui.end()
