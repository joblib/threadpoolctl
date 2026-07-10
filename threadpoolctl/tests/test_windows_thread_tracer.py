import struct
import sys

import pytest

from threadpoolctl._thread_tracer._parsing import (
    EVENT_TRACE_TYPE_DCSTART,
    EVENT_TRACE_TYPE_END,
    EVENT_TRACE_TYPE_START,
    classify_kernel_thread_event,
    parse_kernel_thread_payload,
)
from threadpoolctl._thread_tracer import windows_tracer_available


def _pack_thread_payload(process_id, thread_id):
    return struct.pack("<II", process_id, thread_id)


@pytest.mark.parametrize(
    "opcode,expected",
    [
        (EVENT_TRACE_TYPE_START, "spawn"),
        (EVENT_TRACE_TYPE_DCSTART, "existing"),
        (EVENT_TRACE_TYPE_END, None),
    ],
)
def test_classify_kernel_thread_event(opcode, expected):
    payload = _pack_thread_payload(4242, 99)
    assert classify_kernel_thread_event(opcode, payload, len(payload), 4242) == expected


def test_classify_kernel_thread_event_ignores_other_process():
    payload = _pack_thread_payload(1111, 7)
    assert (
        classify_kernel_thread_event(
            EVENT_TRACE_TYPE_START, payload, len(payload), 4242
        )
        is None
    )


def test_parse_kernel_thread_payload_rejects_short_buffers():
    assert parse_kernel_thread_payload(b"\x01\x02\x03", 3) is None


def test_windows_tracer_available_matches_platform():
    assert windows_tracer_available() == (sys.platform == "win32")
