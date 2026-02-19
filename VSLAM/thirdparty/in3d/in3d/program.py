from typing import Optional
from moderngl_window import resources
from moderngl_window.meta import (
    ProgramDescription,
)

from in3d.logger import setup_logger

logger = setup_logger()


class Program:
    def __init__(
        self,
        path=None,
        vertex_shader=None,
        geometry_shader=None,
        fragment_shader=None,
        tess_control_shader=None,
        tess_evaluation_shader=None,
        defines: Optional[dict] = None,
        varyings: Optional[list[str]] = None,
    ):
        self.program_description = ProgramDescription(
            path=path,
            vertex_shader=vertex_shader,
            geometry_shader=geometry_shader,
            fragment_shader=fragment_shader,
            tess_control_shader=tess_control_shader,
            tess_evaluation_shader=tess_evaluation_shader,
            defines=defines,
            varyings=varyings,
        )

        self.program = self.load_program(self.program_description)
        self.mtime = self.program.extra["meta"].resolved_path.stat().st_mtime

    def load_program(self, program_description: ProgramDescription):
        return resources.programs.load(program_description)

    def maybe_reload(self):
        # TODO don't know what happens if vertex/geometry/fragments are separetely passed in.
        # Probably will fail...
        mtime = self.program.extra["meta"].resolved_path.stat().st_mtime
        if self.mtime < mtime:
            try:
                self.program = self.load_program(self.program_description)
                logger.info("Reloaded programs")
            except Exception as e:
                logger.error(e)
            self.mtime = mtime
