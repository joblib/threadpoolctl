"""Pure-Python helpers to classify kernel thread ETW events."""

import ctypes as ct
import struct

# Kernel thread event opcodes (EVENT_TRACE_TYPE_* for the Thread task).
EVENT_TRACE_TYPE_START = 1
EVENT_TRACE_TYPE_END = 2
EVENT_TRACE_TYPE_DCSTART = 3
EVENT_TRACE_TYPE_DCEND = 4

_MIN_USER_DATA_LENGTH = 8


def parse_kernel_thread_payload(user_data, user_data_length):
    """Extract ``(process_id, thread_id)`` from a kernel Thread event payload.

    The first two fields are stable across Thread event schema versions. Pointer
    fields that follow are ignored because only the identifiers are needed.
    """
    if user_data is None or user_data_length < _MIN_USER_DATA_LENGTH:
        return None

    if isinstance(user_data, (bytes, bytearray)):
        raw = user_data[:user_data_length]
    else:
        address = user_data
        if not isinstance(address, int):
            address = ct.cast(user_data, ct.c_void_p).value or 0
        if not address:
            return None
        raw = ct.string_at(address, user_data_length)

    if len(raw) < _MIN_USER_DATA_LENGTH:
        return None
    process_id, thread_id = struct.unpack_from("<II", raw, 0)
    return process_id, thread_id


def classify_kernel_thread_event(opcode, user_data, user_data_length, target_pid):
    """Return a spawn classification for a kernel Thread event.

    Returns one of:
    - ``"spawn"`` for a thread start event in ``target_pid``
    - ``"existing"`` for a DCStart rundown event in ``target_pid``
    - ``None`` if the event should be ignored
    """
    if opcode not in (
        EVENT_TRACE_TYPE_START,
        EVENT_TRACE_TYPE_END,
        EVENT_TRACE_TYPE_DCSTART,
        EVENT_TRACE_TYPE_DCEND,
    ):
        return None

    parsed = parse_kernel_thread_payload(user_data, user_data_length)
    if parsed is None:
        return None

    process_id, _thread_id = parsed
    if process_id != target_pid:
        return None

    if opcode == EVENT_TRACE_TYPE_START:
        return "spawn"
    if opcode == EVENT_TRACE_TYPE_DCSTART:
        return "existing"
    return None
