import threading


class threadsafe_generator:
    """Takes an generator and makes it thread-safe by
    serializing call to the `next` method of given generator.
    """

    def __init__(self, gen):
        self.gen = gen
        self.lock = threading.Lock()

    def __iter__(self):
        return self.next()

    def next(self):
        with self.lock:
            return self.gen.next()