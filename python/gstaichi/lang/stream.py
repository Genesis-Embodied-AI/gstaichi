"""Platform-agnostic GPU stream and event API for asynchronous execution."""

from gstaichi import _lib
from gstaichi.lang import impl


class Stream:
    """
    Platform-agnostic GPU stream for asynchronous kernel execution.
    
    A stream represents a sequence of operations that execute in order on the GPU.
    Operations on different streams can execute concurrently.
    
    Example::
    
        >>> stream1 = ti.create_stream()
        >>> stream2 = ti.create_stream()
        >>> kernel_a(stream=stream1)  # Runs on stream1
        >>> kernel_b(stream=stream2)  # Runs concurrently on stream2
        >>> ti.sync()  # Wait for all streams
    """
    
    def __init__(self, backend_handle, arch):
        """
        Initialize a Stream wrapper.
        
        Args:
            backend_handle: Native handle (CUstream, hipStream_t, etc.)
            arch: Architecture (ti.cuda, ti.amdgpu, etc.)
        """
        self._handle = backend_handle
        self._arch = arch
    
    @property
    def handle(self):
        """Get the native backend stream handle."""
        return self._handle
    
    def synchronize(self):
        """
        Synchronize this stream (blocks host until all work completes).
        
        This is a host-side synchronization point. For device-side synchronization
        without blocking the host, use events instead.
        """
        _lib.core.gstaichi_python.sync_stream(self._handle)


class Event:
    """
    Platform-agnostic GPU event for synchronization.
    
    Events can be used to create dependencies between streams without blocking
    the host. This enables complex asynchronous execution patterns.
    
    Example::
    
        >>> stream1 = ti.create_stream()
        >>> kernel_a(stream=stream1)
        >>> event = ti.create_event()
        >>> event.record(stream1)  # Record when stream1 completes
        >>> event.wait()  # Make default stream wait (device-side, async from host)
        >>> kernel_b()  # Runs after stream1 completes
    """
    
    def __init__(self, backend_handle, arch):
        """
        Initialize an Event wrapper.
        
        Args:
            backend_handle: Native handle (CUevent, hipEvent_t, etc.)
            arch: Architecture (ti.cuda, ti.amdgpu, etc.)
        """
        self._handle = backend_handle
        self._arch = arch
    
    @property
    def handle(self):
        """Get the native backend event handle."""
        return self._handle
    
    def record(self, stream=None):
        """
        Record this event on a stream.
        
        The event will be signaled when all prior work on the stream completes.
        This is asynchronous from the host's perspective.
        
        Args:
            stream: Stream to record on (None for default stream)
        """
        stream_handle = stream._handle if stream else None
        _lib.core.gstaichi_python.record_event(self._handle, stream_handle)
    
    def wait(self, stream=None):
        """
        Make a stream wait for this event (device-side, non-blocking from host).
        
        The specified stream will not execute further commands until this event
        is signaled. The host continues immediately without blocking.
        
        Args:
            stream: Stream that should wait (None for default stream)
        """
        stream_handle = stream._handle if stream else None
        _lib.core.gstaichi_python.stream_wait_event(stream_handle, self._handle)
    
    def synchronize(self):
        """
        Synchronize with this event (blocks host until event is signaled).
        
        This is a host-side synchronization point.
        """
        _lib.core.gstaichi_python.sync_event(self._handle)


def create_stream():
    """
    Create a new GPU compute stream.
    
    Returns:
        Stream: A new stream for asynchronous execution
        
    Raises:
        RuntimeError: If streams are not supported on the current backend
        
    Example::
    
        >>> stream = ti.create_stream()
        >>> kernel(stream=stream)
    """
    runtime = impl.get_runtime()
    arch = runtime.prog.config().arch
    
    handle = _lib.core.gstaichi_python.create_stream(arch)
    return Stream(handle, arch)


def create_event():
    """
    Create a new GPU synchronization event.
    
    Returns:
        Event: A new event for synchronization
        
    Raises:
        RuntimeError: If events are not supported on the current backend
        
    Example::
    
        >>> event = ti.create_event()
        >>> event.record(stream)
        >>> event.wait()
    """
    runtime = impl.get_runtime()
    arch = runtime.prog.config().arch
    
    handle = _lib.core.gstaichi_python.create_event(arch)
    return Event(handle, arch)


def sync_stream(stream=None):
    """
    Synchronize a stream (blocks host until stream completes).
    
    Args:
        stream: Stream to synchronize (None for default stream)
    """
    stream_handle = stream._handle if stream else None
    _lib.core.gstaichi_python.sync_stream(stream_handle)


def record_event(event, stream=None):
    """
    Record an event on a stream (non-blocking from host).
    
    Args:
        event: Event to record
        stream: Stream to record on (None for default stream)
    """
    stream_handle = stream._handle if stream else None
    _lib.core.gstaichi_python.record_event(event._handle, stream_handle)


def stream_wait_event(stream, event):
    """
    Make a stream wait for an event (device-side, non-blocking from host).
    
    Args:
        stream: Stream that should wait (None for default stream, or Stream object)
        event: Event to wait for
    """
    stream_handle = stream._handle if stream else None
    _lib.core.gstaichi_python.stream_wait_event(stream_handle, event._handle)

