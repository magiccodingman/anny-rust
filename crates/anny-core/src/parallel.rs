//! Where the parallel helpers are allowed to start threads.
//!
//! Spawning is not available everywhere this crate is built. On `wasm32-unknown-unknown` without the
//! `atomics` target feature the `std::thread` API still *compiles* — `thread::scope` and `spawn` are
//! plain functions — but creating a thread panics at run time, so a parallel path that is never
//! exercised natively is a runtime crash in the browser. Every parallel helper therefore asks for its
//! worker count here and takes its sequential path when the answer is one.

/// Whether this target can actually create threads.
pub(crate) const THREADS_SUPPORTED: bool =
    !cfg!(all(target_arch = "wasm32", not(target_feature = "atomics")));

/// Number of workers to use for `work` independent units of work, or `1` to stay sequential.
///
/// Callers must keep their sequential path for the answer `1`; that path is what a browser build runs.
pub(crate) fn worker_threads(work: usize) -> usize {
    if !THREADS_SUPPORTED || work < 2 {
        return 1;
    }
    std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .min(work)
}
