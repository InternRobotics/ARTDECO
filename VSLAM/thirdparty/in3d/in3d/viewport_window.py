import moderngl
import imgui
from in3d.logger import setup_logger
import numpy as np

logger = setup_logger()


class ViewportWindow:
    def __init__(self, name, camera=None):
        self.ctx = moderngl.get_context()
        self.imgui = self.ctx.imgui
        self.pix_ratio = imgui.get_io().display_fb_scale[0]
        self.name = name
        self.camera = camera

        self.size = imgui.Vec2(0, 0)
        self.screen = None
        self.mouse_left_down, self.mouse_right_down = False, False
        self.should_resize = False
        self.has_focus_on_click = False
        self.create_frame_buffer()

    def render(self):
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, imgui.Vec2(0, 0))
        imgui.begin(self.name)
        if not self.handle_resize():
            imgui.pop_style_var()
            imgui.end()
            return

        # use image button to absorb the imgui mouse ctrl
        # Remove styling for the button (so it doesn't blend alpha)
        imgui.push_style_color(imgui.COLOR_BUTTON, 0, 0, 0, 0)
        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0, 0, 0, 0)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0, 0, 0, 0)
        imgui.image_button(
            self.screen.color_attachments[0].glo,
            *self.screen.size,
            uv0=(0, 1),
            uv1=(1, 0),
            frame_padding=0,
        )
        imgui.pop_style_color(3)

        has_focus = imgui.is_item_hovered()
        self.handle_io(has_focus)
        imgui.pop_style_var()
        imgui.end()

    def use(self):
        if self.should_resize and not imgui.get_io().mouse_down[0]:
            self.create_frame_buffer()
            if self.camera:
                self.camera.resize(*self.screen.size)
            self.should_resize = False

        self.screen.use()

    def clear(
        self,
        red=0.0,
        green=0.0,
        blue=0.0,
        alpha=0.0,
        depth=1.0,
        viewport=None,
        color=None,
    ):
        self.screen.clear(red, green, blue, alpha, depth, viewport, color)

    def get_mouse_pos(self):
        # set origin at bottom left
        x, y = imgui.get_mouse_pos()
        wx, wy = imgui.get_cursor_screen_pos()
        return int(x - wx), int(wy - y)

    def handle_io(self, has_focus):
        clicked = imgui.is_mouse_clicked(0) or imgui.is_mouse_clicked(1)
        released = imgui.is_mouse_released(0) or imgui.is_mouse_released(1)
        if clicked and has_focus:
            self.has_focus_on_click = True
        if released:
            self.has_focus_on_click = False

        if self.camera is None:
            return

        left, right = imgui.get_io().mouse_down[0], imgui.get_io().mouse_down[1]
        scroll = imgui.get_io().mouse_wheel * has_focus

        # we want to keep tracking if the drag goes out of focus,
        self.mouse_left_down = (has_focus or self.mouse_left_down) and left
        self.mouse_right_down = (has_focus or self.mouse_right_down) and right

        io_action = self.has_focus_on_click or (has_focus and abs(scroll) > 0)
        if not io_action:
            return
        x, y = self.get_mouse_pos()
        w = 5
        z = np.frombuffer(
            self.screen.read(
                attachment=-1,
                alignment=4,
                components=1,
                dtype="f4",
                viewport=(x - w // 2, y - w // 2, w, w),
            ),
            dtype="f4",
        )
        z = np.min(z)
        io_state = (
            x,
            y,
            z,
            scroll,
            self.mouse_left_down,
            self.mouse_right_down,
            clicked,
            has_focus,
            (0, 0, *self.screen.size),
        )
        self.camera.update(io_state)

    def handle_resize(self):
        view = imgui.get_content_region_available()
        if view.x <= 0 or view.y <= 0:
            self.size = view
            return False

        if view == self.size:
            return True

        self.size = view
        self.should_resize = True
        return True

    def create_frame_buffer(self):
        if self.screen is not None:
            self.imgui.remove_texture(self.screen.color_attachments[0])
            self.screen.release()
            # self.screen.color_attachments[0].release()
            # self.screen.depth_attachment.release()

        size = int(max(self.size.x, 1)), int(max(self.size.y, 1))
        self.screen = self.ctx.framebuffer(
            color_attachments=self.ctx.texture(size, 4),
            depth_attachment=self.ctx.depth_texture(size),
        )
        self.imgui.register_texture(self.screen.color_attachments[0])
