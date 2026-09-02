import time

_enabled = False


def enable():
    global _enabled
    _enabled = True


def log(msg):
    global _enabled
    if _enabled:
        print(msg)


class Measurement:
    def __init__(self):
        self._t_start = 0

    def start(self):
        self._t_start = time.time_ns()

    def stop(self):
        dt = time.time_ns() - self._t_start
        self._t_start = 0
        return dt
