import queue


def try_get_msg(q):
    try:
        msg = q.get_nowait()
    except queue.Empty:
        msg = None
    return msg


class FakeQueue:
    def put(self, arg):
        del arg

    def get_nowait(self):
        raise queue.Empty

    def qsize(self):
        return 0

    def empty(self):
        return True


def new_queue(manager, use_fake=False):
    if use_fake:
        return FakeQueue()
    return manager.Queue()
