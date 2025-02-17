from enum import Enum
import time


class Timer:
    """経過時間測定用のタイマークラス。
    - start/stopを繰り返し呼び出し可能。
    - リセット機能なし。
    """

    class State(Enum):
        STARTED = 0
        STOPPED = 1

    def __init__(self):
        self.start_time = None
        self.state = self.State.STOPPED
        self._elapsed_time = 0.0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    @property
    def is_started(self):
        return self.state == self.State.STARTED

    @property
    def is_stopped(self):
        return self.state == self.State.STOPPED

    @property
    def elapsed_time(self):
        if self.is_started:
            raise RuntimeError('timer must be stopped.')
        return self._elapsed_time

    def start(self):
        if self.is_started:
            raise RuntimeError('timer is already started.')
        self.start_time = time.perf_counter()
        self.state = self.State.STARTED
        return self

    def stop(self):
        if self.is_stopped:
            raise RuntimeError('timer is already stopped.')
        self._elapsed_time += time.perf_counter() - self.start_time
        self.state = self.State.STOPPED
        return self
