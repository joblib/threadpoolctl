"""Private types for the experimental thread spawn tracer."""


class ThreadSpawnStats(object):
    """Statistics collected during a thread tracing window."""

    __slots__ = ("spawn_count", "existing_thread_count", "thread_ids")

    def __init__(self, spawn_count=0, existing_thread_count=0, thread_ids=None):
        self.spawn_count = spawn_count
        self.existing_thread_count = existing_thread_count
        self.thread_ids = frozenset(thread_ids or ())

    def __eq__(self, other):
        if not isinstance(other, ThreadSpawnStats):
            return NotImplemented
        return (
            self.spawn_count == other.spawn_count
            and self.existing_thread_count == other.existing_thread_count
            and self.thread_ids == other.thread_ids
        )

    def __repr__(self):
        return (
            "ThreadSpawnStats(spawn_count={0.spawn_count}, "
            "existing_thread_count={0.existing_thread_count}, "
            "thread_ids={0.thread_ids!r})"
        ).format(self)


class ThreadTracerError(OSError):
    """Raised when the platform tracer cannot start or stop cleanly."""
