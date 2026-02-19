import moderngl


class Image:
    def __init__(self, img=None):
        if img is None:
            # These are overriden when an image is actually written to the image
            height, width, channel = 1, 1, 1
        else:
            height, width = img.shape[:2]
            channel = 1 if img.ndim == 2 else img.shape[2]

        self.ctx = moderngl.get_context()
        self.imgui = self.ctx.imgui
        self.texture = self.register(width, height, channel)
        if img is not None:
            self.texture.write(img.astype("f4").tobytes())

    def register(self, width, height, channel):
        self.width, self.height = width, height
        self.texture = self.ctx.texture((width, height), channel, dtype="f4")
        self.imgui.register_texture(self.texture)
        return self.texture

    def release(self):
        self.imgui.remove_texture(self.texture)
        self.texture.release()

    def write(self, img):
        # check if the shape is correct
        height, width = img.shape[:2]
        channel = 1 if img.ndim == 2 else img.shape[2]
        if (width, height) != self.texture.size or self.texture.components != channel:
            self.release()
            self.texture = self.register(width, height, channel)
        self.texture.write(img.astype("f4").tobytes())
