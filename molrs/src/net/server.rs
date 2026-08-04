//! Native WebSocket server that broadcasts serialized [`Frame`]s to clients.
//!
//! The accept loop and per-client I/O run on a background `std::thread` that
//! owns a multi-thread tokio runtime. The simulation loop stays synchronous:
//! [`FrameServer::send`] never blocks on network writes; when the bounded
//! crossbeam buffer is full the oldest payload is dropped.

use std::io;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use bytes::Bytes;
use crossbeam_channel::{Receiver, Sender, TryRecvError, TrySendError, bounded};
use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpListener;
use tokio::sync::{broadcast, oneshot};
use tokio_tungstenite::tungstenite::Message;

use crate::core::store::frame::Frame;
use crate::stream::{MessageFormat, StreamError, frame_to_bytes};

use super::message::ControlCommand;

/// Configuration for a [`FrameServer`].
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Wire encoding for outbound frames (default: MessagePack).
    pub format: MessageFormat,
    /// Capacity of the simulation→network frame buffer (default: 4).
    ///
    /// When full, [`FrameServer::send`] drops the oldest buffered frame so the
    /// producer never blocks.
    pub buffer_size: usize,
    /// Reserved maximum stream rate in Hz. Not enforced in v1 (no-op).
    pub max_frame_rate: f64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            format: MessageFormat::MessagePack,
            buffer_size: 4,
            max_frame_rate: 0.0,
        }
    }
}

/// Error returned by [`FrameServer::send`].
#[derive(Debug)]
pub enum SendError {
    /// Frame encoding failed.
    Encode(StreamError),
    /// The server has been shut down or the background thread exited.
    Closed,
}

impl std::fmt::Display for SendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SendError::Encode(e) => write!(f, "frame encode error: {e}"),
            SendError::Closed => write!(f, "frame server closed"),
        }
    }
}

impl std::error::Error for SendError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SendError::Encode(e) => Some(e),
            SendError::Closed => None,
        }
    }
}

impl From<StreamError> for SendError {
    fn from(e: StreamError) -> Self {
        SendError::Encode(e)
    }
}

struct Shared {
    format: MessageFormat,
    /// Simulation → background bridge (payload already encoded).
    frame_tx: Mutex<Option<Sender<Bytes>>>,
    /// Competing receiver used only to drop the oldest frame when full.
    drop_rx: Mutex<Option<Receiver<Bytes>>>,
    cmd_rx: Receiver<ControlCommand>,
    client_count: Arc<AtomicUsize>,
    local_addr: SocketAddr,
    shutting_down: AtomicBool,
    join: Mutex<Option<JoinHandle<()>>>,
    shutdown_tx: Mutex<Option<oneshot::Sender<()>>>,
}

/// Broadcasts serialized [`Frame`]s to WebSocket clients and collects control commands.
///
/// Clone freely: all clones share the same server state. Call
/// [`FrameServer::shutdown`] to stop the background thread, or drop the last
/// clone to join on exit.
#[derive(Clone)]
pub struct FrameServer {
    shared: Arc<Shared>,
}

impl FrameServer {
    /// Bind a WebSocket server on `addr` with default [`ServerConfig`].
    ///
    /// Use `"127.0.0.1:0"` to pick an ephemeral port, then read
    /// [`local_addr`](Self::local_addr).
    pub fn bind(addr: impl Into<String>) -> io::Result<Self> {
        Self::bind_with(addr, ServerConfig::default())
    }

    /// Bind a WebSocket server on `addr` with the given configuration.
    pub fn bind_with(addr: impl Into<String>, config: ServerConfig) -> io::Result<Self> {
        let addr = addr.into();
        let buffer_size = config.buffer_size.max(1);
        let format = config.format;

        let (frame_tx, frame_rx) = bounded::<Bytes>(buffer_size);
        // Second receiver competes for messages so send() can free a slot when full.
        let drop_rx = frame_rx.clone();
        let (cmd_tx, cmd_rx) = bounded::<ControlCommand>(64);
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<io::Result<SocketAddr>>();
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

        let client_count = Arc::new(AtomicUsize::new(0));
        let client_count_thread = Arc::clone(&client_count);

        let join = std::thread::Builder::new()
            .name("molrs-frame-server".into())
            .spawn(move || {
                let rt = match tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .worker_threads(2)
                    .thread_name("molrs-net")
                    .build()
                {
                    Ok(rt) => rt,
                    Err(e) => {
                        let _ = ready_tx.send(Err(io::Error::other(e)));
                        return;
                    }
                };

                rt.block_on(async move {
                    let listener = match TcpListener::bind(&addr).await {
                        Ok(l) => l,
                        Err(e) => {
                            let _ = ready_tx.send(Err(e));
                            return;
                        }
                    };
                    let local = match listener.local_addr() {
                        Ok(a) => a,
                        Err(e) => {
                            let _ = ready_tx.send(Err(e));
                            return;
                        }
                    };
                    let _ = ready_tx.send(Ok(local));

                    run_server(
                        listener,
                        frame_rx,
                        cmd_tx,
                        client_count_thread,
                        format,
                        shutdown_rx,
                    )
                    .await;
                });
            })?;

        let local_addr = ready_rx
            .recv()
            .map_err(|_| io::Error::other("frame server thread exited before bind"))??;

        Ok(FrameServer {
            shared: Arc::new(Shared {
                format,
                frame_tx: Mutex::new(Some(frame_tx)),
                drop_rx: Mutex::new(Some(drop_rx)),
                cmd_rx,
                client_count,
                local_addr,
                shutting_down: AtomicBool::new(false),
                join: Mutex::new(Some(join)),
                shutdown_tx: Mutex::new(Some(shutdown_tx)),
            }),
        })
    }

    /// Local socket address the server is listening on.
    pub fn local_addr(&self) -> SocketAddr {
        self.shared.local_addr
    }

    /// Number of currently connected WebSocket clients.
    pub fn client_count(&self) -> usize {
        self.shared.client_count.load(Ordering::Relaxed)
    }

    /// Encode `frame` and enqueue it for broadcast.
    ///
    /// Never blocks on network I/O. If the internal buffer is full, the oldest
    /// pending frame is dropped so this call returns promptly.
    pub fn send(&self, frame: &Frame) -> Result<(), SendError> {
        let bytes = frame_to_bytes(frame, self.shared.format)?;
        self.send_bytes(Bytes::from(bytes))
    }

    /// Async convenience wrapper around [`send`](Self::send).
    pub async fn send_async(&self, frame: &Frame) -> Result<(), SendError> {
        self.send(frame)
    }

    /// Wait for the next control command from any client.
    ///
    /// Returns `None` when the server is shutting down or the command channel
    /// is closed. Polls with a short sleep so it is safe to call from any tokio
    /// runtime (not only the server's background runtime).
    pub async fn recv_command(&self) -> Option<ControlCommand> {
        loop {
            match self.shared.cmd_rx.try_recv() {
                Ok(cmd) => return Some(cmd),
                Err(TryRecvError::Disconnected) => return None,
                Err(TryRecvError::Empty) => {
                    if self.shared.shutting_down.load(Ordering::Acquire) {
                        return None;
                    }
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            }
        }
    }

    /// Like [`recv_command`](Self::recv_command) but returns `None` if `timeout` elapses first.
    pub async fn recv_command_timeout(&self, timeout: Duration) -> Option<ControlCommand> {
        tokio::time::timeout(timeout, self.recv_command())
            .await
            .unwrap_or_default()
    }

    /// Signal the accept loop to stop and join the background thread.
    pub fn shutdown(self) {
        self.request_shutdown();
        if let Ok(mut guard) = self.shared.join.lock()
            && let Some(handle) = guard.take()
        {
            let _ = handle.join();
        }
    }

    fn request_shutdown(&self) {
        self.shared.shutting_down.store(true, Ordering::Release);
        if let Ok(mut guard) = self.shared.frame_tx.lock() {
            *guard = None;
        }
        if let Ok(mut guard) = self.shared.drop_rx.lock() {
            *guard = None;
        }
        if let Ok(mut guard) = self.shared.shutdown_tx.lock()
            && let Some(tx) = guard.take()
        {
            let _ = tx.send(());
        }
    }

    fn send_bytes(&self, bytes: Bytes) -> Result<(), SendError> {
        let tx_guard = self.shared.frame_tx.lock().map_err(|_| SendError::Closed)?;
        let tx = tx_guard.as_ref().ok_or(SendError::Closed)?;

        let mut item = bytes;
        // Bound the spin so a theoretical race cannot hang the simulation.
        for _ in 0..64 {
            match tx.try_send(item) {
                Ok(()) => return Ok(()),
                Err(TrySendError::Disconnected(_)) => return Err(SendError::Closed),
                Err(TrySendError::Full(v)) => {
                    if let Ok(drop_guard) = self.shared.drop_rx.lock()
                        && let Some(drop_rx) = drop_guard.as_ref()
                    {
                        let _ = drop_rx.try_recv();
                    }
                    item = v;
                }
            }
        }
        // Last attempt: drop the new frame rather than block.
        Ok(())
    }
}

impl Drop for FrameServer {
    fn drop(&mut self) {
        // Only the last Arc clone should join; earlier clones leave the server running.
        if Arc::strong_count(&self.shared) > 1 {
            return;
        }
        self.request_shutdown();
        if let Ok(mut guard) = self.shared.join.lock()
            && let Some(handle) = guard.take()
        {
            let _ = handle.join();
        }
    }
}

// --- Background runtime -------------------------------------------------------

async fn run_server(
    listener: TcpListener,
    frame_rx: Receiver<Bytes>,
    cmd_tx: Sender<ControlCommand>,
    client_count: Arc<AtomicUsize>,
    format: MessageFormat,
    mut shutdown_rx: oneshot::Receiver<()>,
) {
    let (bcast_tx, _) = broadcast::channel::<Bytes>(16);

    // Bridge crossbeam → tokio broadcast on a blocking task.
    let bcast_bridge = bcast_tx.clone();
    let bridge = tokio::task::spawn_blocking(move || {
        while let Ok(payload) = frame_rx.recv() {
            // Ignore "no receivers" — normal when no clients are connected.
            let _ = bcast_bridge.send(payload);
        }
    });

    loop {
        tokio::select! {
            _ = &mut shutdown_rx => break,
            accepted = listener.accept() => {
                let Ok((stream, _)) = accepted else { continue };
                let Ok(ws) = tokio_tungstenite::accept_async(stream).await else {
                    continue;
                };
                client_count.fetch_add(1, Ordering::Relaxed);
                let bcast_rx = bcast_tx.subscribe();
                let cmd_tx = cmd_tx.clone();
                let client_count = Arc::clone(&client_count);
                tokio::spawn(async move {
                    handle_client(ws, bcast_rx, cmd_tx, format).await;
                    client_count.fetch_sub(1, Ordering::Relaxed);
                });
            }
        }
    }

    drop(bcast_tx);
    let _ = bridge.await;
}

type WsStream = tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>;

async fn handle_client(
    ws: WsStream,
    mut bcast_rx: broadcast::Receiver<Bytes>,
    cmd_tx: Sender<ControlCommand>,
    format: MessageFormat,
) {
    let (mut write, mut read) = ws.split();

    loop {
        tokio::select! {
            frame = bcast_rx.recv() => {
                match frame {
                    Ok(payload) => {
                        let msg = match format {
                            MessageFormat::MessagePack => Message::Binary(payload),
                            MessageFormat::Json => {
                                match String::from_utf8(payload.to_vec()) {
                                    Ok(s) => Message::Text(s.into()),
                                    Err(_) => continue,
                                }
                            }
                        };
                        if write.send(msg).await.is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
            incoming = read.next() => {
                match incoming {
                    Some(Ok(Message::Text(text))) => {
                        if let Ok(cmd) = serde_json::from_str::<ControlCommand>(text.as_ref()) {
                            let _ = cmd_tx.try_send(cmd);
                        }
                    }
                    Some(Ok(Message::Binary(bin))) => {
                        if let Ok(cmd) = rmp_serde::from_slice::<ControlCommand>(bin.as_ref()) {
                            let _ = cmd_tx.try_send(cmd);
                        }
                    }
                    Some(Ok(Message::Ping(p))) => {
                        let _ = write.send(Message::Pong(p)).await;
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    Some(Ok(_)) => {}
                    Some(Err(_)) => break,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::store::block::Block;
    use crate::stream::bytes_to_frame;
    use crate::types::{F, I};
    use futures_util::{SinkExt, StreamExt};
    use ndarray::Array1;
    use tokio_tungstenite::connect_async;

    fn sample_frame(tag: i32) -> Frame {
        let mut atoms = Block::new();
        atoms
            .insert(
                "x",
                Array1::from_vec(vec![1.0 as F, 2.0 as F, tag as F]).into_dyn(),
            )
            .unwrap();
        atoms
            .insert(
                "y",
                Array1::from_vec(vec![0.0 as F, 1.0 as F, 2.0 as F]).into_dyn(),
            )
            .unwrap();
        atoms
            .insert(
                "z",
                Array1::from_vec(vec![0.0 as F, 0.0 as F, 1.0 as F]).into_dyn(),
            )
            .unwrap();
        atoms
            .insert(
                "serial",
                Array1::from_vec(vec![1 as I, 2 as I, 3 as I]).into_dyn(),
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.meta.insert("tag", tag);
        frame
    }

    async fn wait_clients(server: &FrameServer, n: usize) {
        for _ in 0..100 {
            if server.client_count() == n {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!(
            "timed out waiting for {n} clients (have {})",
            server.client_count()
        );
    }

    #[tokio::test]
    async fn bind_and_client_connects() {
        let server = FrameServer::bind("127.0.0.1:0").expect("bind");
        let url = format!("ws://{}", server.local_addr());

        let (mut ws, _) = connect_async(&url).await.expect("connect");
        wait_clients(&server, 1).await;
        assert_eq!(server.client_count(), 1);

        ws.close(None).await.ok();
        server.shutdown();
    }

    #[tokio::test]
    async fn send_frame_received_by_client() {
        let server = FrameServer::bind("127.0.0.1:0").expect("bind");
        let url = format!("ws://{}", server.local_addr());
        let (mut ws, _) = connect_async(&url).await.expect("connect");
        wait_clients(&server, 1).await;

        let frame = sample_frame(7);
        server.send(&frame).expect("send");

        let msg = tokio::time::timeout(Duration::from_secs(2), ws.next())
            .await
            .expect("timeout")
            .expect("stream closed")
            .expect("ws error");

        let bytes = match msg {
            Message::Binary(b) => b.to_vec(),
            Message::Text(t) => t.as_bytes().to_vec(),
            other => panic!("unexpected message: {other:?}"),
        };
        let decoded = bytes_to_frame(&bytes, MessageFormat::MessagePack).expect("decode");
        assert!(decoded.contains_key("atoms"));
        let x = decoded["atoms"].get_float("x").unwrap();
        assert!((x[2] - 7.0).abs() < 1e-12);

        ws.close(None).await.ok();
        server.shutdown();
    }

    #[tokio::test]
    async fn client_pause_command_received() {
        let server = FrameServer::bind("127.0.0.1:0").expect("bind");
        let url = format!("ws://{}", server.local_addr());
        let (mut ws, _) = connect_async(&url).await.expect("connect");
        wait_clients(&server, 1).await;

        let payload = serde_json::to_string(&ControlCommand::Pause).unwrap();
        ws.send(Message::Text(payload.into()))
            .await
            .expect("send cmd");

        let cmd = server
            .recv_command_timeout(Duration::from_secs(2))
            .await
            .expect("expected Pause");
        assert_eq!(cmd, ControlCommand::Pause);

        ws.close(None).await.ok();
        server.shutdown();
    }

    #[tokio::test]
    async fn buffer_full_send_does_not_block() {
        let server = FrameServer::bind_with(
            "127.0.0.1:0",
            ServerConfig {
                format: MessageFormat::MessagePack,
                buffer_size: 1,
                max_frame_rate: 0.0,
            },
        )
        .expect("bind");

        let start = std::time::Instant::now();
        for i in 0..3 {
            server.send(&sample_frame(i)).expect("send");
        }
        assert!(
            start.elapsed() < Duration::from_secs(1),
            "send blocked for {:?}",
            start.elapsed()
        );
        server.shutdown();
    }

    #[tokio::test]
    async fn buffer_full_client_eventually_gets_latest() {
        let server = FrameServer::bind_with(
            "127.0.0.1:0",
            ServerConfig {
                format: MessageFormat::MessagePack,
                buffer_size: 1,
                max_frame_rate: 0.0,
            },
        )
        .expect("bind");
        let url = format!("ws://{}", server.local_addr());
        let (mut ws, _) = connect_async(&url).await.expect("connect");
        wait_clients(&server, 1).await;

        for i in 1..=3 {
            server.send(&sample_frame(i)).expect("send");
        }

        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        let mut saw_latest = false;
        while tokio::time::Instant::now() < deadline {
            match tokio::time::timeout(Duration::from_millis(200), ws.next()).await {
                Ok(Some(Ok(Message::Binary(b)))) => {
                    if let Ok(decoded) = bytes_to_frame(b.as_ref(), MessageFormat::MessagePack)
                        && let Some(x) = decoded.get("atoms").and_then(|a| a.get_float("x"))
                        && (x[2] - 3.0).abs() < 1e-12
                    {
                        saw_latest = true;
                        break;
                    }
                }
                Ok(Some(Ok(_))) => {}
                Ok(Some(Err(_))) | Ok(None) => break,
                Err(_) => {}
            }
        }
        assert!(saw_latest, "client never received the latest frame (tag 3)");

        ws.close(None).await.ok();
        server.shutdown();
    }
}
