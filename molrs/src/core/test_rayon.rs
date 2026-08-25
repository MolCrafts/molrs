//! Test-only rayon global pool: bounded workers, env override, installed once.
//!
//! `cargo test` runs many tests at once. The first `par_iter` in each of them
//! would otherwise try to spawn `available_parallelism()` workers into rayon's
//! **global** pool. Concurrent `build_global` races (`GlobalPoolAlreadyInitialized`)
//! and `pthread_create` returning `EAGAIN` (`WouldBlock`) leave the pool
//! uninitialised — every later `par_iter` then panics.
//!
//! Tests install a small pool **before** any parallel iterator. Default is 2
//! workers so the parallel path still runs. Override with `MOLRS_TEST_THREADS`
//! (clamped to 2..=8).

use rayon::prelude::*;
use std::sync::Once;

/// Environment variable that sets the test rayon worker count.
pub(crate) const THREADS_ENV: &str = "MOLRS_TEST_THREADS";

const DEFAULT_THREADS: usize = 2;
const MIN_THREADS: usize = 2;
const MAX_THREADS: usize = 8;

/// Worker count for the test pool: `MOLRS_TEST_THREADS`, else 2, clamped to 2..=8.
pub(crate) fn worker_count() -> usize {
    std::env::var(THREADS_ENV)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_THREADS)
        .clamp(MIN_THREADS, MAX_THREADS)
}

/// Install the test pool once. Safe to call from every rayon-using test.
pub(crate) fn ensure() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        let n = worker_count();
        // AlreadyInitialized: another test or rust-analyzer won the race after
        // we picked `n`. The pool exists; thread count may not match `n`.
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global();
    });
}

#[test]
fn test_pool_is_multithreaded() {
    ensure();
    let n = rayon::current_num_threads();
    assert!(
        n >= MIN_THREADS,
        "rayon test pool must be multithreaded (got {n} workers; \
         set {THREADS_ENV}>={MIN_THREADS})"
    );

    use std::collections::HashSet;
    use std::sync::Mutex;
    let seen = Mutex::new(HashSet::new());
    (0..1024).into_par_iter().for_each(|_| {
        if let Some(i) = rayon::current_thread_index() {
            seen.lock().unwrap().insert(i);
        }
    });
    let workers = seen.lock().unwrap().len();
    assert!(
        workers >= MIN_THREADS,
        "par_iter used {workers} worker(s); the parallel path was not tested"
    );
}

#[test]
fn test_thread_count_follows_env_or_default() {
    ensure();
    let configured = worker_count();
    assert!((MIN_THREADS..=MAX_THREADS).contains(&configured));
    // If we won the install, current_num_threads matches. If rust-analyzer
    // already built a larger pool, we still have at least MIN_THREADS.
    assert!(rayon::current_num_threads() >= MIN_THREADS);
}
