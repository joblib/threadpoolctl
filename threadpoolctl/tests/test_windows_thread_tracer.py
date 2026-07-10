import ctypes as ct
import struct
import sys

import pytest

from threadpoolctl._thread_tracer import (
    ThreadSpawnStats,
    ThreadTracerError,
    WindowsThreadSpawnTracer,
    windows_tracer_available,
)
from threadpoolctl._thread_tracer._parsing import (
    EVENT_TRACE_TYPE_DCSTART,
    EVENT_TRACE_TYPE_END,
    EVENT_TRACE_TYPE_START,
    classify_kernel_thread_event,
    parse_kernel_thread_payload,
)


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


def test_windows_tracer_start_requires_windows(monkeypatch):
    if sys.platform == "win32":
        pytest.skip("Platform-specific negative test")

    monkeypatch.setattr(
        "threadpoolctl._thread_tracer._windows_etw.sys.platform",
        "linux",
    )
    monkeypatch.setattr(
        "threadpoolctl._thread_tracer._windows_etw._advapi32",
        None,
    )
    monkeypatch.setattr(
        "threadpoolctl._thread_tracer._windows_etw.StartTraceW",
        None,
    )

    tracer = WindowsThreadSpawnTracer(1234)
    with pytest.raises(ThreadTracerError, match="only supported on Windows"):
        tracer.start()


def test_windows_tracer_lifecycle_with_mocked_etw(monkeypatch):
    from threadpoolctl._thread_tracer import _windows_etw as etw

    monkeypatch.setattr(etw, "_advapi32", object())
    monkeypatch.setattr(etw.sys, "platform", "win32")

    spawn_payload = _pack_thread_payload(9001, 12)
    existing_payload = _pack_thread_payload(9001, 34)

    def _make_record(opcode, payload):
        record = etw.EVENT_RECORD()
        record.EventHeader.EventDescriptor.Opcode = opcode
        record.UserDataLength = len(payload)
        buffer = ct.create_string_buffer(payload)
        record._payload_buffer = buffer
        record.UserData = ct.cast(buffer, ct.c_void_p)
        return ct.pointer(record)

    events = [
        _make_record(etw.EVENT_TRACE_TYPE_DCSTART, existing_payload),
        _make_record(etw.EVENT_TRACE_TYPE_START, spawn_payload),
        _make_record(etw.EVENT_TRACE_TYPE_START, spawn_payload),
    ]
    event_state = {"index": 0, "callback": None, "stop": False}

    def fake_start_trace(session_handle, logger_name, props):
        session_handle._obj.value = 1
        return etw.ERROR_SUCCESS

    def fake_open_trace(trace_logfile):
        event_state["callback"] = trace_logfile._obj.EventRecordCallback
        return etw.TRACEHANDLE(2)

    def fake_process_trace(trace_handle, count, start_time, end_time):
        if event_state["stop"]:
            return 1
        callback = event_state["callback"]
        while event_state["index"] < len(events):
            callback(events[event_state["index"]])
            event_state["index"] += 1
        return 1

    def fake_close_trace(trace_handle):
        event_state["stop"] = True
        return etw.ERROR_SUCCESS

    def fake_control_trace(session_handle, logger_name, props, control_code):
        return etw.ERROR_SUCCESS

    monkeypatch.setattr(etw, "StartTraceW", fake_start_trace)
    monkeypatch.setattr(etw, "OpenTraceW", fake_open_trace)
    monkeypatch.setattr(etw, "ProcessTrace", fake_process_trace)
    monkeypatch.setattr(etw, "CloseTrace", fake_close_trace)
    monkeypatch.setattr(etw, "ControlTraceW", fake_control_trace)

    tracer = WindowsThreadSpawnTracer(9001, include_existing_threads=True)
    tracer.start()
    stats = tracer.stop()

    assert stats == ThreadSpawnStats(
        spawn_count=2,
        existing_thread_count=1,
        thread_ids=frozenset({12, 34}),
    )


def test_windows_tracer_reports_existing_kernel_logger_session(monkeypatch):
    from threadpoolctl._thread_tracer import _windows_etw as etw

    monkeypatch.setattr(etw, "_advapi32", object())
    monkeypatch.setattr(etw.sys, "platform", "win32")

    def fake_start_trace(session_handle, logger_name, props):
        return etw.ERROR_ALREADY_EXISTS

    monkeypatch.setattr(etw, "StartTraceW", fake_start_trace)

    tracer = WindowsThreadSpawnTracer(1234)
    with pytest.raises(ThreadTracerError, match="already in use"):
        tracer.start()
