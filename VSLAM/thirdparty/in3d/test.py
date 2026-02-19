import glfw
import moderngl
import platform

# Configuration
program_name = "GLFW window"
window_width = 1200
window_height = 800
background_r = 0.1
background_g = 0.3
background_b = 0.2


class App:
    def __init__(self, width, height, title, gl_version=(3, 3)):
        # Initialize GLFW
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        else:
            print("GLFW initialized")

        # Configure GLFW
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.DEPTH_BITS, 24)
        glfw.window_hint(glfw.STENCIL_BITS, 8)
        glfw.window_hint(glfw.DOUBLEBUFFER, 1)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, gl_version[0])
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, gl_version[1])
        if platform.system() == "Darwin":  # macOS
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)  # Required on macOS
            # Avoid frame buffer size from doubling on retina display
            glfw.window_hint(glfw.COCOA_RETINA_FRAMEBUFFER, glfw.FALSE)
        # elif platform.system() == "Windows" or platform.system() == "Linux":

        # Handle high DPI scaling
        monitor = glfw.get_primary_monitor()
        xscale, yscale = glfw.get_monitor_content_scale(monitor)
        print(f"Monitor content scale: xscale={xscale}, yscale={yscale}")
        if xscale > 1 or yscale > 1:
            high_dpi_scale_factor = xscale  # Use xscale as the scale factor
            glfw.window_hint(glfw.SCALE_TO_MONITOR, glfw.TRUE)

        # Create window
        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        # Make the OpenGL context current
        glfw.make_context_current(self.window)
        glfw.set_framebuffer_size_callback(self.window, self.framebuffer_size_callback)
        glfw.swap_interval(1)  # Enable VSync

        # Initialize ModernGL context
        self.ctx = moderngl.create_context()
        self.ctx.viewport = (0, 0, width, height)
        self.clear_color = (background_r, background_g, background_b, 1.0)

    def framebuffer_size_callback(self, window, width, height):
        """Callback to adjust viewport when window size changes."""
        self.ctx.viewport = (0, 0, width, height)

    def render(self):
        """Rendering loop."""
        while not glfw.window_should_close(self.window):
            # Clear the screen with the background color
            self.ctx.clear(*self.clear_color)

            # Swap buffers and poll events
            glfw.swap_buffers(self.window)
            glfw.poll_events()

            window_width, window_height = glfw.get_window_size(self.window)
            framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(
                self.window
            )

            print(f"Window size: {window_width}x{window_height}")
            print(f"Framebuffer size: {framebuffer_width}x{framebuffer_height}")

    def terminate(self):
        """Clean up resources."""
        if self.window:
            glfw.destroy_window(self.window)
        glfw.terminate()

    def run(self):
        """Run the application."""
        try:
            self.render()
        finally:
            self.terminate()


if __name__ == "__main__":
    app = App(window_width, window_height, program_name)
    app.run()
