//! The shape rules of the GPU entry points, exercised without a device.
//!
//! `check_shapes` is the single place where caller-supplied dimensions are believed or rejected, and
//! it is deliberately free of wgpu: that is what lets these cases run on machines with no adapter at
//! all — including CI — where the device-level tests in `parity.rs` skip. A malformed call must come
//! back as `GpuError::InvalidInput`; trusting it instead is what ends in a panicking slice or a wgpu
//! validation failure after the dispatch has already been built.

use anny_gpu::blendshapes::check_shapes;
use anny_gpu::GpuError;

/// Unwraps the message of an `InvalidInput`, failing loudly on any other outcome.
fn message(result: Result<usize, GpuError>) -> String {
    match result {
        Err(GpuError::InvalidInput(m)) => m,
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

#[test]
fn valid_shapes_are_accepted_and_report_the_output_length() {
    let batch = 4;
    let c = 624;
    let size = 41_154;
    assert_eq!(
        check_shapes(batch, c, size, c * size, batch * c).expect("valid"),
        batch * size
    );
    assert_eq!(check_shapes(1, 1, 1, 1, 1).expect("valid"), 1);
}

#[test]
fn zero_batch_coefficient_count_or_output_size_is_rejected() {
    assert!(message(check_shapes(0, 624, 8, 624 * 8, 0)).contains("nonzero"));
    assert!(message(check_shapes(1, 0, 8, 0, 0)).contains("nonzero"));
    assert!(message(check_shapes(1, 624, 0, 0, 624)).contains("nonzero"));
}

#[test]
fn wrong_coefficient_length_is_rejected() {
    let m = message(check_shapes(4, 624, 8, 624 * 8, 4 * 624 - 3));
    assert!(m.contains("expected 2496 coefficients"), "{m}");
    assert!(m.contains("got 2493"), "{m}");
}

#[test]
fn wrong_blendshape_length_is_rejected() {
    let m = message(check_shapes(1, 4, 8, 4 * 8 + 1, 4));
    assert!(m.contains("expected 32 blendshape values"), "{m}");
    assert!(m.contains("got 33"), "{m}");
}

#[test]
fn overflowing_shapes_are_rejected_rather_than_wrapping() {
    // The blendshape shape cannot be represented.
    assert!(message(check_shapes(1, usize::MAX, 2, 0, 1)).contains("overflow"));
    // The coefficient count cannot be represented.
    assert!(message(check_shapes(usize::MAX, 2, 1, 2, usize::MAX)).contains("overflow"));
    // Both inputs are individually consistent, and the output length still cannot be represented.
    assert!(message(check_shapes(usize::MAX, 1, 2, 2, usize::MAX)).contains("overflow"));
}
