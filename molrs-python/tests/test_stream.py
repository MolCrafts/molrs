"""FFI-seam tests for `molrs.stream` — live Frame streaming.

Depth of the streaming semantics (bounded-buffer drop, broadcast fan-out,
graceful shutdown) lives in the Rust unit tests. What is checked here is the
seam: that the Python types exist, carry their payloads across the boundary
intact, and fail loudly instead of guessing.
"""

import socket
import threading

import numpy as np
import pytest

import molrs.io as mio
import molrs
from molrs.stream import ControlCommand

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _frame(n: int = 3, offset: float = 0.0) -> molrs.Frame:
    return molrs.Frame(
        blocks={
            "atoms": {
                "x": np.arange(n, dtype=np.float64) + offset,
                "y": np.zeros(n, dtype=np.float64),
                "z": np.zeros(n, dtype=np.float64),
                "element": ["C"] * n,
            }
        }
    )


class TestFrameWireCodec:
    def test_round_trip_preserves_columns(self):
        frame = _frame()
        back = mio.read_frame_bytes(mio.write_frame_bytes(frame))
        np.testing.assert_allclose(back["atoms"]["x"], frame["atoms"]["x"])
        assert list(back["atoms"]["element"]) == ["C", "C", "C"]

    def test_round_trip_returns_the_rich_frame_subclass(self):
        # `_lib.Frame.from_bytes` is a staticmethod on the bare PyO3 core; the
        # rich layer shadows it. Without the shadow a decoded stream frame
        # would not accept `frame["atoms"]["x"]`.
        back = mio.read_frame_bytes(mio.write_frame_bytes(_frame()))
        assert isinstance(back, molrs.Frame)

    def test_round_trip_preserves_the_box(self):
        frame = _frame()
        frame.box = molrs.Box(np.eye(3) * 10.0)
        back = mio.read_frame_bytes(mio.write_frame_bytes(frame))
        assert back.box is not None
        np.testing.assert_allclose(back.box.matrix, frame.box.matrix)

    @pytest.mark.parametrize("fmt", ["msgpack", "json"])
    def test_both_formats_round_trip(self, fmt):
        back = mio.read_frame_bytes(mio.write_frame_bytes(_frame(), fmt), fmt)
        np.testing.assert_allclose(back["atoms"]["x"], [0.0, 1.0, 2.0])

    def test_decoding_with_the_wrong_format_raises(self):
        # Silently reading MessagePack as JSON is how a stream turns into
        # garbage columns instead of an error.
        payload = mio.write_frame_bytes(_frame(), "msgpack")
        with pytest.raises(ValueError):
            mio.read_frame_bytes(payload, "json")

    def test_unknown_format_name_raises(self):
        with pytest.raises(ValueError, match="unknown wire format"):
            mio.write_frame_bytes(_frame(), "messagepack")


class TestControlCommand:
    @pytest.mark.parametrize(
        ("command", "kind"),
        [
            (ControlCommand.pause(), "pause"),
            (ControlCommand.resume(), "resume"),
            (ControlCommand.set_frame_rate(30.0), "set_frame_rate"),
            (ControlCommand.set_subset([3, 1, 2]), "set_subset"),
            (ControlCommand.request_key_frame(), "request_key_frame"),
        ],
    )
    def test_kind_matches_the_wire_tag(self, command, kind):
        assert command.kind == kind

    def test_payload_attributes_are_none_off_their_own_kind(self):
        assert ControlCommand.pause().hz is None
        assert ControlCommand.pause().atom_ids is None
        assert ControlCommand.set_frame_rate(12.5).hz == 12.5
        assert ControlCommand.set_frame_rate(12.5).atom_ids is None

    def test_subset_preserves_order(self):
        assert ControlCommand.set_subset([9, 1, 7]).atom_ids == [9, 1, 7]

    @pytest.mark.parametrize("fmt", ["json", "msgpack"])
    def test_round_trip(self, fmt):
        original = ControlCommand.set_subset([4, 2])
        assert ControlCommand.from_bytes(original.to_bytes(fmt), fmt) == original

    def test_json_wire_shape_is_the_documented_tagged_object(self):
        # A browser sends `{"type": "pause"}` by hand; if the tag drifts the
        # producer stops seeing commands with no error anywhere.
        import json

        assert json.loads(ControlCommand.pause().to_bytes("json")) == {"type": "pause"}

    @pytest.mark.parametrize("hz", [0.0, -1.0, float("nan"), float("inf")])
    def test_nonsense_frame_rate_raises(self, hz):
        with pytest.raises(ValueError):
            ControlCommand.set_frame_rate(hz)


@pytest.mark.skipif(
    not hasattr(molrs.stream, "FramePublisher"),
    reason="FramePublisher is native-only (absent on Pyodide)",
)
class TestFrameServer:
    def test_ephemeral_bind_reports_its_port(self):
        with molrs.stream.FramePublisher("127.0.0.1:0") as server:
            host, _, port = server.address.rpartition(":")
            assert host == "127.0.0.1"
            assert int(port) > 0

    def test_starts_with_no_clients(self):
        with molrs.stream.FramePublisher("127.0.0.1:0") as server:
            assert server.client_count == 0

    def test_send_without_a_client_does_not_block(self):
        # The bounded buffer drops rather than stalls; a producer must be able
        # to stream into the void. buffer_size=1 forces the drop path.
        with molrs.stream.FramePublisher("127.0.0.1:0", buffer_size=1) as server:
            done = threading.Event()

            def produce():
                for i in range(20):
                    server.send(_frame(offset=float(i)))
                done.set()

            thread = threading.Thread(target=produce)
            thread.start()
            thread.join(timeout=10.0)
            assert done.is_set(), "send() blocked with no reader attached"

    def test_recv_command_polls_without_blocking(self):
        with molrs.stream.FramePublisher("127.0.0.1:0") as server:
            assert server.recv_command() is None

    def test_recv_command_honours_a_timeout(self):
        import time

        with molrs.stream.FramePublisher("127.0.0.1:0") as server:
            start = time.monotonic()
            assert server.recv_command(timeout=0.05) is None
            assert time.monotonic() - start >= 0.04

    @pytest.mark.parametrize("timeout", [-1.0, float("nan")])
    def test_nonsense_timeout_raises(self, timeout):
        with molrs.stream.FramePublisher("127.0.0.1:0") as server:
            with pytest.raises(ValueError):
                server.recv_command(timeout=timeout)

    def test_zero_buffer_size_raises(self):
        with pytest.raises(ValueError, match="buffer_size"):
            molrs.stream.FramePublisher("127.0.0.1:0", buffer_size=0)

    def test_unknown_format_raises_before_binding(self):
        with pytest.raises(ValueError, match="unknown wire format"):
            molrs.stream.FramePublisher("127.0.0.1:0", format="protobuf")

    def test_bad_address_raises_ioerror(self):
        with pytest.raises(OSError):
            molrs.stream.FramePublisher("256.256.256.256:1")

    def test_use_after_close_raises(self):
        server = molrs.stream.FramePublisher("127.0.0.1:0")
        server.close()
        with pytest.raises(ValueError, match="closed"):
            server.send(_frame())

    def test_close_is_idempotent(self):
        server = molrs.stream.FramePublisher("127.0.0.1:0")
        server.close()
        server.close()

    def test_port_is_released_after_close(self):
        # A leaked listener thread would keep the port bound and turn the next
        # bind in a long-running session into a confusing EADDRINUSE.
        server = molrs.stream.FramePublisher("127.0.0.1:0")
        host, _, port = server.address.rpartition(":")
        server.close()
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((host, int(port)))
