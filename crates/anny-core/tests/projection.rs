use anny_core::math::*;

#[test]
fn near_block_diagonal_covariance_has_a_smooth_proper_rotation() {
    // Reduced real spine02 covariance. The fixed-size nalgebra 3x3 SVD
    // produced finite-difference jumps here despite a unique, well-conditioned
    // polar factor. This test does not alter step size to hide those jumps.
    let a = Mat3::from_column_slice(&[
        0.008082683754625297,
        -1.0829814561304923e-12,
        1.3503822761601683e-18,
        -1.910694570768888e-18,
        -0.004105052101308156,
        0.011252090431984541,
        9.269716816514249e-20,
        -0.009519426434884801,
        0.007024097500744112,
    ]);
    let da = Mat3::from_column_slice(&[
        0.0008826227194450331,
        7.21989749046996e-14,
        5.79317596362738e-19,
        6.088049308390284e-22,
        -0.0003111806934047451,
        0.0024132682546082296,
        1.6865881817711213e-19,
        -0.0009020783768580006,
        0.001590564586251317,
    ]);
    let r = special_procrustes(&a);
    let s = (r.transpose() * a + a.transpose() * r) * 0.5;
    let k = Mat3::identity() * s.trace() - s;
    let b = r.transpose() * da - da.transpose() * r;
    let w = k.try_inverse().unwrap() * Vec3::new(b[(2, 1)], b[(0, 2)], b[(1, 0)]);
    let skew = Mat3::new(0., -w[2], w[1], w[2], 0., -w[0], -w[1], w[0], 0.);
    for h in [1e-3, 1e-4, 1e-5, 1e-6] {
        let numeric =
            (special_procrustes(&(a + h * da)) - special_procrustes(&(a - h * da))) / (2. * h);
        let error = (numeric - r * skew).norm();
        assert!(error < 2e-6, "polar derivative jump at h={h}: {error}");
    }
    assert!((r.transpose() * r - Mat3::identity()).norm() < 1e-12);
    assert!((r.determinant() - 1.).abs() < 1e-12);
}

#[test]
fn proper_projection_handles_reflections_rank_two_and_extreme_scales() {
    let r = rotvec(&Vec3::new(0.3, -0.2, 0.1));
    for last in [0.5, 0., -0.5] {
        let m = r * Mat3::from_diagonal(&Vec3::new(4., 2., last));
        for scale in [1., 1e-200, 1e200] {
            let actual = special_procrustes(&(m * scale));
            assert!((actual - r).norm() < 1e-12, "last={last}, scale={scale}");
            assert!((actual.determinant() - 1.).abs() < 1e-12);
        }
    }
    assert_eq!(special_procrustes(&Mat3::zeros()), Mat3::identity());
    // Rank one has multiple optimal proper rotations. Test its objective, not
    // a fabricated unique orientation or a particular SVD null-space choice.
    let rank_one = r * Mat3::from_diagonal(&Vec3::new(4., 0., 0.));
    let actual = special_procrustes(&rank_one);
    assert!((actual.determinant() - 1.).abs() < 1e-12);
    assert!(((actual.transpose() * rank_one).trace() - 4.).abs() < 1e-12);
}
