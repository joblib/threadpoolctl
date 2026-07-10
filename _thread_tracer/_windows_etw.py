"""Windows kernel ETW tracer for counting thread spawn events.

This is a private prototype built with ctypes only (no compiled extensions).
It uses the ``NT Kernel Logger`` session with ``EVENT_TRACE_FLAG_THREAD`` and
counts Thread Start / DCStart events for a target process PID.

Administrator privileges are typically required.
"""

import ctypes as ct
import ctypes.wintypes as wt
import sys
import threading
import time
import uuid

from _thread_tracer._parsing import (
    EVENT_TRACE_TYPE_DCSTART,
    EVENT_TRACE_TYPE_START,
    classify_kernel_thread_event,
    parse_kernel_thread_payload,
)
from _thread_tracer._types import ThreadSpawnStats, ThreadTracerError

ERROR_SUCCESS = 0
ERROR_ALREADY_EXISTS = 183

KERNEL_LOGGER_NAME = "NT Kernel Logger"
SYSTEM_TRACE_PROVIDER_GUID = "{9E814AAD-3204-11D2-9A82-006008A86939}"

EVENT_TRACE_FLAG_THREAD = 0x00000002
EVENT_TRACE_REAL_TIME_MODE = 0x00000100
EVENT_TRACE_SYSTEM_LOGGER_MODE = 0x02000000
EVENT_TRACE_CONTROL_STOP = 1

PROCESS_TRACE_MODE_REAL_TIME = 0x00000100
PROCESS_TRACE_MODE_EVENT_RECORD = 0x10000000

WNODE_FLAG_TRACED_GUID = 0x00020000

TRACEHANDLE = ct.c_ulonglong
INVALID_PROCESSTRACE_HANDLE = TRACEHANDLE(-1)


class GUID(ct.Structure):
    _fields_ = [
        ("Data1", ct.c_uint32),
        ("Data2", ct.c_uint16),
        ("Data3", ct.c_uint16),
        ("Data4", ct.c_ubyte * 8),
    ]

    @classmethod
    def from_string(cls, guid_string):
        guid = cls()
        parts = guid_string.strip("{}").split("-")
        guid.Data1 = int(parts[0], 16)
        guid.Data2 = int(parts[1], 16)
        guid.Data3 = int(parts[2], 16)
        hex_data4 = parts[3] + parts[4]
        for index in range(8):
            guid.Data4[index] = int(hex_data4[index * 2 : index * 2 + 2], 16)
        return guid


class WNODE_HEADER(ct.Structure):
    _fields_ = [
        ("BufferSize", ct.c_ulong),
        ("ProviderId", ct.c_ulong),
        ("HistoricalContext", ct.c_uint64),
        ("TimeStamp", wt.LARGE_INTEGER),
        ("Guid", GUID),
        ("ClientContext", ct.c_ulong),
        ("Flags", ct.c_ulong),
    ]


class EVENT_DESCRIPTOR(ct.Structure):
    _fields_ = [
        ("Id", ct.c_ushort),
        ("Version", ct.c_ubyte),
        ("Channel", ct.c_ubyte),
        ("Level", ct.c_ubyte),
        ("Opcode", ct.c_ubyte),
        ("Task", ct.c_ushort),
        ("Keyword", ct.c_uint64),
    ]


class EVENT_HEADER(ct.Structure):
    _fields_ = [
        ("Size", ct.c_ushort),
        ("HeaderType", ct.c_ushort),
        ("Flags", ct.c_ushort),
        ("EventProperty", ct.c_ushort),
        ("ThreadId", ct.c_ulong),
        ("ProcessId", ct.c_ulong),
        ("TimeStamp", wt.LARGE_INTEGER),
        ("ProviderId", GUID),
        ("EventDescriptor", EVENT_DESCRIPTOR),
        ("KernelTime", ct.c_ulong),
        ("UserTime", ct.c_ulong),
        ("ActivityId", GUID),
    ]


class ETW_BUFFER_CONTEXT(ct.Structure):
    _fields_ = [
        ("ProcessorNumber", ct.c_ubyte),
        ("Alignment", ct.c_ubyte),
        ("LoggerId", ct.c_ushort),
    ]


class EVENT_RECORD(ct.Structure):
    _fields_ = [
        ("EventHeader", EVENT_HEADER),
        ("BufferContext", ETW_BUFFER_CONTEXT),
        ("ExtendedDataCount", ct.c_ushort),
        ("UserDataLength", ct.c_ushort),
        ("ExtendedData", ct.c_void_p),
        ("UserData", ct.c_void_p),
        ("UserContext", ct.c_void_p),
    ]


class EVENT_TRACE_PROPERTIES(ct.Structure):
    _fields_ = [
        ("Wnode", WNODE_HEADER),
        ("BufferSize", ct.c_ulong),
        ("MinimumBuffers", ct.c_ulong),
        ("MaximumBuffers", ct.c_ulong),
        ("MaximumFileSize", ct.c_ulong),
        ("LogFileMode", ct.c_ulong),
        ("FlushTimer", ct.c_ulong),
        ("EnableFlags", ct.c_ulong),
        ("AgeLimit", ct.c_ulong),
        ("NumberOfBuffers", ct.c_ulong),
        ("FreeBuffers", ct.c_ulong),
        ("EventsLost", ct.c_ulong),
        ("BuffersWritten", ct.c_ulong),
        ("LogBuffersLost", ct.c_ulong),
        ("RealTimeBuffersLost", ct.c_ulong),
        ("LoggerThreadId", wt.HANDLE),
        ("LogFileNameOffset", ct.c_ulong),
        ("LoggerNameOffset", ct.c_ulong),
    ]


class EVENT_TRACE_LOGFILE(ct.Structure):
    pass


if sys.platform == "win32":
    EVENT_RECORD_CALLBACK = ct.WINFUNCTYPE(None, ct.POINTER(EVENT_RECORD))
else:
    EVENT_RECORD_CALLBACK = ct.CFUNCTYPE(None, ct.POINTER(EVENT_RECORD))

EVENT_TRACE_LOGFILE._fields_ = [
    ("LogFileName", ct.c_wchar_p),
    ("LoggerName", ct.c_wchar_p),
    ("CurrentTime", ct.c_longlong),
    ("BuffersRead", ct.c_ulong),
    ("ProcessTraceMode", ct.c_ulong),
    ("CurrentEvent", ct.c_void_p),
    ("LogfileHeader", ct.c_void_p),
    ("BufferCallback", ct.c_void_p),
    ("BufferSize", ct.c_ulong),
    ("Filled", ct.c_ulong),
    ("EventsLost", ct.c_ulong),
    ("EventRecordCallback", EVENT_RECORD_CALLBACK),
    ("IsKernelTrace", ct.c_ulong),
    ("Context", ct.c_void_p),
]


_advapi32 = None
StartTraceW = None
ControlTraceW = None
OpenTraceW = None
ProcessTrace = None
CloseTrace = None


def _ensure_advapi32():
    global _advapi32, StartTraceW, ControlTraceW, OpenTraceW, ProcessTrace, CloseTrace
    if _advapi32 is not None:
        return _advapi32

    if sys.platform != "win32":
        raise ThreadTracerError("Windows ETW tracer is only supported on Windows")

    _advapi32 = ct.windll.advapi32

    StartTraceW = _advapi32.StartTraceW
    StartTraceW.argtypes = [
        ct.POINTER(TRACEHANDLE),
        ct.c_wchar_p,
        ct.POINTER(EVENT_TRACE_PROPERTIES),
    ]
    StartTraceW.restype = ct.c_ulong

    ControlTraceW = _advapi32.ControlTraceW
    ControlTraceW.argtypes = [
        TRACEHANDLE,
        ct.c_wchar_p,
        ct.POINTER(EVENT_TRACE_PROPERTIES),
        ct.c_ulong,
    ]
    ControlTraceW.restype = ct.c_ulong

    OpenTraceW = _advapi32.OpenTraceW
    OpenTraceW.argtypes = [ct.POINTER(EVENT_TRACE_LOGFILE)]
    OpenTraceW.restype = TRACEHANDLE

    ProcessTrace = _advapi32.ProcessTrace
    ProcessTrace.argtypes = [
        ct.POINTER(TRACEHANDLE),
        ct.c_ulong,
        ct.c_void_p,
        ct.c_void_p,
    ]
    ProcessTrace.restype = ct.c_ulong

    CloseTrace = _advapi32.CloseTrace
    CloseTrace.argtypes = [TRACEHANDLE]
    CloseTrace.restype = ct.c_ulong
    return _advapi32


def _make_trace_properties(enable_flags):
    max_str_len = 1024
    buf_size = (
        ct.sizeof(EVENT_TRACE_PROPERTIES) + 2 * ct.sizeof(ct.c_wchar) * max_str_len
    )
    buf = (ct.c_char * buf_size)()
    props = ct.cast(ct.pointer(buf), ct.POINTER(EVENT_TRACE_PROPERTIES))

    props.contents.Wnode.BufferSize = buf_size
    props.contents.Wnode.Flags = WNODE_FLAG_TRACED_GUID
    props.contents.Wnode.Guid = GUID.from_string(SYSTEM_TRACE_PROVIDER_GUID)
    props.contents.BufferSize = 64
    props.contents.LogFileMode = (
        EVENT_TRACE_REAL_TIME_MODE | EVENT_TRACE_SYSTEM_LOGGER_MODE
    )
    props.contents.EnableFlags = enable_flags
    props.contents.LoggerNameOffset = ct.sizeof(EVENT_TRACE_PROPERTIES)
    return buf, props


class _TraceStats(object):
    def __init__(self):
        self._lock = threading.Lock()
        self.spawn_count = 0
        self.existing_thread_count = 0
        self._thread_ids = set()

    def record(self, classification, thread_id):
        with self._lock:
            if classification == "spawn":
                self.spawn_count += 1
            elif classification == "existing":
                self.existing_thread_count += 1
            else:
                return
            self._thread_ids.add(thread_id)

    def snapshot(self):
        with self._lock:
            return ThreadSpawnStats(
                spawn_count=self.spawn_count,
                existing_thread_count=self.existing_thread_count,
                thread_ids=set(self._thread_ids),
            )


class WindowsThreadSpawnTracer(object):
    """Count thread spawn events for a process using kernel ETW.

    Parameters
    ----------
    pid : int
        Target process identifier.
    include_existing_threads : bool, default=False
        If True, also count DCStart rundown events emitted for threads that
        were already alive when tracing started.
    """

    def __init__(self, pid, include_existing_threads=False):
        self.pid = int(pid)
        self.include_existing_threads = include_existing_threads
        self._stats = _TraceStats()
        self._session_handle = TRACEHANDLE(0)
        self._trace_handle = TRACEHANDLE(0)
        self._trace_properties_buf = None
        self._trace_properties = None
        self._trace_logfile = None
        self._callback = None
        self._consumer_thread = None
        self._stop_event = threading.Event()
        self._started = False
        self._session_started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False

    def start(self):
        if self._started:
            raise ThreadTracerError("Tracer is already started")

        _ensure_advapi32()
        self._trace_properties_buf, self._trace_properties = _make_trace_properties(
            EVENT_TRACE_FLAG_THREAD
        )
        status = StartTraceW(
            ct.byref(self._session_handle),
            KERNEL_LOGGER_NAME,
            self._trace_properties,
        )
        if status == ERROR_ALREADY_EXISTS:
            raise ThreadTracerError(
                "The NT Kernel Logger session is already in use. Stop the "
                "existing session or run with a single tracer instance."
            )
        if status != ERROR_SUCCESS:
            raise ThreadTracerError(status, ct.FormatError(status))

        self._session_started = True
        self._install_consumer()
        self._started = True

    def stop(self):
        if not self._started:
            return self._stats.snapshot()

        self._stop_event.set()
        time.sleep(0.2)
        if self._trace_handle.value:
            CloseTrace(self._trace_handle)
        if self._consumer_thread is not None:
            self._consumer_thread.join(timeout=10)
            self._consumer_thread = None

        if self._session_started:
            status = ControlTraceW(
                self._session_handle,
                KERNEL_LOGGER_NAME,
                self._trace_properties,
                EVENT_TRACE_CONTROL_STOP,
            )
            if status != ERROR_SUCCESS:
                raise ThreadTracerError(status, ct.FormatError(status))
            self._session_started = False

        self._started = False
        return self._stats.snapshot()

    def _install_consumer(self):
        self._trace_logfile = EVENT_TRACE_LOGFILE()
        self._trace_logfile.LoggerName = KERNEL_LOGGER_NAME
        self._trace_logfile.ProcessTraceMode = (
            PROCESS_TRACE_MODE_REAL_TIME | PROCESS_TRACE_MODE_EVENT_RECORD
        )
        self._trace_logfile.Context = ct.c_void_p(id(self))
        self._callback = EVENT_RECORD_CALLBACK(self._on_event_record)
        self._trace_logfile.EventRecordCallback = self._callback

        self._trace_handle = OpenTraceW(ct.byref(self._trace_logfile))
        if self._trace_handle == INVALID_PROCESSTRACE_HANDLE:
            raise ThreadTracerError(ct.get_last_error(), "OpenTraceW failed")

        self._consumer_thread = threading.Thread(
            target=self._consume_trace,
            name="threadpoolctl-etw-consumer-{0}".format(uuid.uuid4().hex[:8]),
        )
        self._consumer_thread.daemon = True
        self._consumer_thread.start()

    def _consume_trace(self):
        trace_handle = self._trace_handle
        while not self._stop_event.is_set():
            status = ProcessTrace(ct.byref(trace_handle), 1, None, None)
            if status != ERROR_SUCCESS:
                break
            if self._stop_event.is_set():
                break

    def _on_event_record(self, record_pointer):
        record = record_pointer.contents
        opcode = record.EventHeader.EventDescriptor.Opcode
        if opcode not in (EVENT_TRACE_TYPE_START, EVENT_TRACE_TYPE_DCSTART):
            return
        if opcode == EVENT_TRACE_TYPE_DCSTART and not self.include_existing_threads:
            return

        classification = classify_kernel_thread_event(
            opcode,
            record.UserData,
            record.UserDataLength,
            self.pid,
        )
        if classification is None:
            return

        parsed = parse_kernel_thread_payload(record.UserData, record.UserDataLength)
        if parsed is None:
            return
        _process_id, thread_id = parsed
        self._stats.record(classification, thread_id)
