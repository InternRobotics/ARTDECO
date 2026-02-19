import logging
from collections import deque

import imgui


class MaxCapacityHandler(logging.Handler):
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity
        self.records = deque(maxlen=capacity)

    def emit(self, record):
        # print(record)
        self.records.append(record)

    def get_records(self):
        return list(self.records)


def setup_logger(name="default_logger", level=logging.DEBUG, capacity=128):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if logger has handlers to avoid adding multiple handlers
    if not any(isinstance(handler, MaxCapacityHandler) for handler in logger.handlers):
        capacity_handler = MaxCapacityHandler(capacity=capacity)
        capacity_handler.setLevel(level)
        logger.addHandler(capacity_handler)

    return logger


def imgui_render_log(logger, FPS=None):
    if FPS is None:
        imgui.begin("Log")
    else:
        imgui.begin(f"Log (FPS: {FPS})###Log")
    capacity_handler = logger.handlers[0]
    records = capacity_handler.get_records()
    level2color = {
        logging.DEBUG: (0.0, 1.0, 0.0, 1.0),
        logging.INFO: (1.0, 1.0, 1.0, 1.0),
        logging.WARNING: (1.0, 1.0, 0.0, 1.0),
        logging.ERROR: (1.0, 0.0, 0.0, 1.0),
        logging.CRITICAL: (1.0, 0.0, 1.0, 1.0),
    }
    for record in records:
        imgui.text_colored(
            f"[{record.levelname}] {record.msg}", *level2color[record.levelno]
        )
    if imgui.get_scroll_y() >= imgui.get_scroll_max_y():
        imgui.set_scroll_here_y(1.0)
    imgui.end()
