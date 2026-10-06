"""Live Frame streaming — ``molrs::stream``.

A producer binds a :class:`Publisher` and calls ``send(frame)`` once per
simulation step. The call never blocks on the network: frames go through a
bounded buffer that drops the oldest payload when a viewer cannot keep up, so
a slow client slows nothing down. Viewers dial the socket and decode payloads
with :func:`read_frame_bytes`; :func:`write_frame_bytes` is its inverse, the
encoding the publisher puts on the wire (``"msgpack"`` or ``"json"``).

Traffic the other way is :class:`ControlCommand` — a viewer asking the producer
to pause, change rate, or restrict the streamed atom subset. Nothing here acts
on a command; the producer decides what it means.

:class:`Publisher` binds a TCP listener and is therefore native-only, gated
exactly as ``molrs::stream::publisher`` is. A Pyodide build has
:class:`ControlCommand` and no server — absent, rather than a stub that would
fail only once someone tried to connect.

Example
-------
Producer::

    import molrs

    with molrs.stream.Publisher("127.0.0.1:8765") as server:
        for _ in range(n_steps):
            integrator.step()
            server.send(integrator.frame)
            cmd = server.recv_command()
            if cmd is not None and cmd.kind == "pause":
                ...
"""

from ._lib import ControlCommand, read_frame_bytes, write_frame_bytes

__all__ = ["ControlCommand", "read_frame_bytes", "write_frame_bytes"]

try:  # native only — see the module docstring
    from ._lib import Publisher  # noqa: F401 — appended to __all__ below
except ImportError:  # pragma: no cover — Pyodide build
    pass
else:
    __all__.append("Publisher")
