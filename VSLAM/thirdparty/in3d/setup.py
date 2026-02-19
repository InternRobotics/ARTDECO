from pathlib import Path
from setuptools import setup

thirdparty_path = Path(__file__).parent / "thirdparty"
pyimgui_path = (thirdparty_path / "pyimgui").as_uri()

setup(
    install_requires=[
        f"imgui @ {pyimgui_path}",
        "moderngl==5.12.0",
        "moderngl-window==2.4.6",
        "glfw",
        "pyglm",
        "msgpack",
        "numpy",
        "matplotlib",
        "trimesh[easy]",
    ]
)
