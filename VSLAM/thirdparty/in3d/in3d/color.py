import matplotlib
import matplotlib.cm
import numpy as np


def hex2rgba(hex, alpha=1.0):
    hex = hex.lstrip("#")
    rgb = tuple(int(hex[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return rgb + (alpha,)


def depth2rgb(depth, min=None, max=None, colormap="jet"):
    dtype = depth.dtype
    dmin = np.nanmin(depth) if min is None else min
    dmax = np.nanmax(depth) if max is None else max
    d = (depth - dmin) / np.maximum((dmax - dmin), 1e-8)
    d = np.clip(d, 0, 1)
    colormap = matplotlib.cm.get_cmap(colormap)
    rgb = colormap(d)[..., :3]

    return np.ascontiguousarray(rgb.astype(dtype))


def gray2rgb(gray):
    gray = np.clip(gray, 0, 1)
    gray = gray.reshape(*gray.shape, 1)
    return gray.repeat(3, axis=-1)
