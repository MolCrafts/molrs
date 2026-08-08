"""Live Frame streaming — ``molrs::stream``.

A producer binds a :class:`FramePublisher` and calls ``send(frame)`` once per
simulation step. The call never blocks on the network: frames go through a
bounded buffer that drops the oldest payload when a viewer cannot keep up, so
a slow client slows nothing down. Viewers dial the socket and decode payloads
with :func:`molrs.io.read_frame_bytes`.

Traffic the other way is :class:`ControlCommand` — a viewer asking the producer
to pause, change rate, or restrict the streamed atom subset. Nothing here acts
on a command; the producer decides what it means.

:class:`FramePublisher` binds a TCP listener and is therefore native-only, gated
exactly as ``molrs::stream::publisher`` is. A Pyodide build has
:class:`ControlCommand` and no server — absent, rather than a stub that would
fail only once someone tried to connect.

Example
-------
Producer::

    import molrs

    with molrs.stream.FramePublisher("127.0.0.1:8765") as server:
        for _ in range(n_steps):
            integrator.step()
            server.send(integrator.frame)
            cmd = server.recv_command()
            if cmd is not None and cmd.kind == "pause":
                ...
"""

from ._lib import ControlCommand as ControlCommand

__all__ = ["ControlCommand"]

try:  # native only — see the module docstring
    from ._lib import FramePublisher as FramePublisher
except ImportError:  # pragma: no cover — Pyodide build
    pass
else:
    __all__.append("FramePublisher")
