// Port of NAVER Anny numerical conventions, Copyright (C) 2025 NAVER Corp.
// SPDX-License-Identifier: Apache-2.0
use crate::{ensure, Error, Result};
use nalgebra::{Matrix3, Matrix4, UnitQuaternion, Vector3};
pub type Vec3 = Vector3<Scalar>;
pub type Mat3 = Matrix3<Scalar>;
pub type Mat4 = Matrix4<Scalar>;

pub fn vec3(x: &[Scalar]) -> Vec3 {
    Vec3::new(x[0], x[1], x[2])
}
pub fn mat3(x: &[Scalar]) -> Mat3 {
    Mat3::from_row_slice(x)
}
pub fn mat4(x: &[Scalar]) -> Mat4 {
    Mat4::from_row_slice(x)
}
pub fn write3(m: &Mat3, out: &mut [Scalar]) {
    for i in 0..3 {
        for j in 0..3 {
            out[i * 3 + j] = m[(i, j)];
        }
    }
}
pub fn write4(m: &Mat4, out: &mut [Scalar]) {
    for i in 0..4 {
        for j in 0..4 {
            out[i * 4 + j] = m[(i, j)];
        }
    }
}
pub fn rotation(m: &Mat4) -> Mat3 {
    m.fixed_view::<3, 3>(0, 0).into_owned()
}
pub fn translation(m: &Mat4) -> Vec3 {
    m.fixed_view::<3, 1>(0, 3).into_owned()
}
pub fn rigid(r: &Mat3, t: &Vec3) -> Mat4 {
    let mut h = Mat4::identity();
    h.fixed_view_mut::<3, 3>(0, 0).copy_from(r);
    h.fixed_view_mut::<3, 1>(0, 3).copy_from(t);
    h
}
pub fn inverse_rigid(m: &Mat4) -> Mat4 {
    let r = rotation(m).transpose();
    rigid(&r, &(-r * translation(m)))
}
pub fn point(m: &Mat4, v: &Vec3) -> Vec3 {
    rotation(m) * v + translation(m)
}
pub fn rotvec(v: &Vec3) -> Mat3 {
    let theta = v.norm();
    if theta < 1e-6 {
        return Mat3::new(1., -v[2], v[1], v[2], 1., -v[0], -v[1], v[0], 1.);
    }
    let k = *v / theta;
    let s = theta.sin();
    let c = 1. - theta.cos();
    let (x, y, z) = (k[0], k[1], k[2]);
    Mat3::new(
        1. - y * y * c - z * z * c,
        x * y * c - z * s,
        x * z * c + y * s,
        x * y * c + z * s,
        1. - x * x * c - z * z * c,
        -x * s + y * z * c,
        x * z * c - y * s,
        x * s + y * z * c,
        1. - x * x * c - y * y * c,
    )
}
/// Rig JSON roll angles are evaluated in float32 upstream, then widened.
#[allow(clippy::unnecessary_cast)] // Scalar is f64 in reference mode, f32 in typed mode.
pub fn legacy_roll_y(roll: Scalar) -> Mat3 {
    let r = roll as f32;
    let x = (r * r).sqrt() / 2.;
    let sinc = if x.abs() < 1e-3 {
        1. - x * x / 6. + (x * x) * (x * x) / 120.
    } else {
        x.sin() / x
    };
    let y = (sinc / 2.) * r;
    let w = x.cos();
    let y2 = y * y;
    let w2 = w * w;
    let d = w2 - y2;
    let s = 2. * (y * w);
    Mat3::new(
        d as Scalar,
        0.,
        s as Scalar,
        0.,
        (y2 + w2) as Scalar,
        0.,
        -s as Scalar,
        0.,
        d as Scalar,
    )
}
/// Closest proper rotation, including reflection correction.
///
/// Maximize tr(R^T M) via the largest eigenvector of the symmetric quaternion
/// matrix. A scaled cyclic Jacobi solve avoids forming M^T M and the small
/// discontinuities of the fixed-size SVD on nearly block-diagonal covariances.
/// This matters for directional derivatives, not just the forward residual.
pub fn special_procrustes(m: &Mat3) -> Mat3 {
    let scale = m.amax();
    if scale == 0. {
        return Mat3::identity();
    }
    let a = m / scale;
    let (xx, yy, zz) = (a[(0, 0)], a[(1, 1)], a[(2, 2)]);
    // Quaternion order is x, y, z, w. The quadratic form is tr(R(q)^T M).
    let mut k = Mat4::new(
        xx - yy - zz,
        a[(0, 1)] + a[(1, 0)],
        a[(0, 2)] + a[(2, 0)],
        a[(2, 1)] - a[(1, 2)],
        a[(0, 1)] + a[(1, 0)],
        yy - xx - zz,
        a[(1, 2)] + a[(2, 1)],
        a[(0, 2)] - a[(2, 0)],
        a[(0, 2)] + a[(2, 0)],
        a[(1, 2)] + a[(2, 1)],
        zz - xx - yy,
        a[(1, 0)] - a[(0, 1)],
        a[(2, 1)] - a[(1, 2)],
        a[(0, 2)] - a[(2, 0)],
        a[(1, 0)] - a[(0, 1)],
        xx + yy + zz,
    );
    let mut vectors = Mat4::identity();
    for _ in 0..32 {
        let mut changed = false;
        for p in 0..3 {
            for q in p + 1..4 {
                let off = k[(p, q)];
                if off.abs() <= Scalar::EPSILON * k.amax() {
                    continue;
                }
                changed = true;
                let tau = (k[(q, q)] - k[(p, p)]) / (2. * off);
                let t = tau.signum() / (tau.abs() + tau.hypot(1.));
                let c = 1. / (1. + t * t).sqrt();
                let s = t * c;
                k[(p, p)] -= t * off;
                k[(q, q)] += t * off;
                k[(p, q)] = 0.;
                k[(q, p)] = 0.;
                for i in 0..4 {
                    if i != p && i != q {
                        let x = k[(i, p)];
                        let y = k[(i, q)];
                        k[(i, p)] = c * x - s * y;
                        k[(p, i)] = k[(i, p)];
                        k[(i, q)] = s * x + c * y;
                        k[(q, i)] = k[(i, q)];
                    }
                    let x = vectors[(i, p)];
                    let y = vectors[(i, q)];
                    vectors[(i, p)] = c * x - s * y;
                    vectors[(i, q)] = s * x + c * y;
                }
            }
        }
        if !changed {
            break;
        }
    }
    let mut largest = 0;
    for i in 1..4 {
        if k[(i, i)] > k[(largest, largest)] {
            largest = i;
        }
    }
    let q = vectors.column(largest).normalize();
    UnitQuaternion::new_normalize(nalgebra::Quaternion::new(q[3], q[0], q[1], q[2]))
        .to_rotation_matrix()
        .into_inner()
}
pub fn linear_interpolation(
    value: Scalar,
    anchors: &[Scalar],
    extrapolate: bool,
) -> Result<Vec<Scalar>> {
    ensure(
        value.is_finite()
            && anchors.len() >= 2
            && anchors.iter().all(|v| v.is_finite())
            && anchors.windows(2).all(|w| w[1] > w[0]),
        "invalid interpolation value or anchors",
    )?;
    let i = anchors
        .partition_point(|&a| a < value)
        .clamp(1, anchors.len() - 1);
    let mut a = (value - anchors[i - 1]) / (anchors[i] - anchors[i - 1]);
    if !extrapolate {
        a = a.clamp(0., 1.);
    }
    let mut w = vec![0.; anchors.len()];
    w[i - 1] = 1. - a;
    w[i] = a;
    Ok(w)
}
pub fn interpolate(
    value: Scalar,
    anchors: &[Scalar],
    values: &[Scalar],
    extrapolate: bool,
) -> Result<Scalar> {
    ensure(anchors.len() == values.len(), "interpolation size mismatch")?;
    Ok(linear_interpolation(value, anchors, extrapolate)?
        .iter()
        .zip(values)
        .map(|(a, b)| a * b)
        .sum())
}
pub fn propagation_order(parents: &[i32]) -> Result<Vec<usize>> {
    ensure(!parents.is_empty(), "empty skeleton")?;
    for &p in parents {
        ensure(
            p >= -1 && p < (parents.len() as i32),
            "parent index outside skeleton",
        )?;
    }
    let mut seen = vec![false; parents.len()];
    let mut order = Vec::new();
    let mut level: Vec<_> = parents
        .iter()
        .enumerate()
        .filter(|(_, p)| **p == -1)
        .map(|(i, _)| i)
        .collect();
    while !level.is_empty() {
        for &i in &level {
            seen[i] = true;
            order.push(i);
        }
        level = (0..parents.len())
            .filter(|&i| !seen[i] && parents[i] >= 0 && seen[parents[i] as usize])
            .collect();
    }
    ensure(order.len() == parents.len(), "cyclic or rootless skeleton")?;
    Ok(order)
}
pub fn forward_kinematics(
    parents: &[i32],
    rest: &[Mat4],
    delta: &[Mat4],
    base: Option<&Mat4>,
) -> Result<(Vec<Mat4>, Vec<Mat4>)> {
    ensure(
        parents.len() == rest.len() && rest.len() == delta.len(),
        "kinematics size mismatch",
    )?;
    let mut poses = vec![Mat4::identity(); rest.len()];
    let mut transforms = poses.clone();
    for i in propagation_order(parents)? {
        let t = rest[i] * delta[i];
        poses[i] = if parents[i] < 0 {
            base.map_or(t, |b| b * t)
        } else {
            transforms[parents[i] as usize] * t
        };
        transforms[i] = poses[i] * inverse_rigid(&rest[i]);
    }
    Ok((poses, transforms))
}
pub fn absolute_kinematics(
    parents: &[i32],
    rest: &[Mat4],
    orient: &[Mat3],
    base: Option<&Mat4>,
) -> Result<(Vec<Mat4>, Vec<Mat4>)> {
    ensure(
        parents.len() == rest.len() && rest.len() == orient.len(),
        "absolute kinematics size mismatch",
    )?;
    let mut poses = vec![Mat4::identity(); rest.len()];
    let mut transforms = poses.clone();
    for i in propagation_order(parents)? {
        let t = if parents[i] < 0 {
            base.map_or(rest[i], |b| b * rest[i])
        } else {
            transforms[parents[i] as usize] * rest[i]
        };
        poses[i] = rigid(&orient[i], &translation(&t));
        transforms[i] = poses[i] * inverse_rigid(&rest[i]);
    }
    Ok((poses, transforms))
}
pub fn tail_pose(head: &Vec3, tail: &Vec3, roll: &Mat3) -> Mat4 {
    let y = (tail - head).normalize();
    let cross = y.cross(&Vec3::y());
    let norm = cross.norm();
    let axis = cross / norm;
    let angle = norm.atan2(y[1]);
    // Preserve the upstream fallback, including its exactly-parallel convention.
    let r = if (axis.norm_squared() - 1.).abs() < 0.1 {
        rotvec(&(-angle * axis))
    } else {
        Mat3::from_diagonal(&Vec3::new(1., -1., -1.))
    };
    rigid(&(r * roll), head)
}
pub fn shortest_arc(target: &Vec3, source: &Vec3) -> Mat3 {
    let a = *target / target.norm().max(1e-8);
    let b = *source / source.norm().max(1e-8);
    let dot = a.dot(&b).clamp(-1., 1.);
    if dot < -1. + 1e-6 {
        let axis = b
            .cross(&if b[0].abs() > 0.6 {
                Vec3::y()
            } else {
                Vec3::x()
            })
            .normalize();
        return 2. * axis * axis.transpose() - Mat3::identity();
    }
    let v = b.cross(&a);
    let s = Mat3::new(0., -v[2], v[1], v[2], 0., -v[0], -v[1], v[0], 0.);
    Mat3::identity() + s + s * s / (1. + dot)
}
pub fn qmul(a: [Scalar; 4], b: [Scalar; 4]) -> [Scalar; 4] {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}
pub fn quaternion(r: &Mat3) -> [Scalar; 4] {
    let q = UnitQuaternion::from_rotation_matrix(&nalgebra::Rotation3::from_matrix_unchecked(*r));
    let q = q.quaternion();
    [q.i, q.j, q.k, q.w]
}
pub fn dual_quaternion(h: &Mat4) -> ([Scalar; 4], [Scalar; 4]) {
    let q = quaternion(&rotation(h));
    let t = translation(h);
    // Upstream deliberately uses scalar part 1 (not 0) in this factor.
    (q, qmul([0.5 * t[0], 0.5 * t[1], 0.5 * t[2], 1.], q))
}
pub fn dqs_point(
    v: &Vec3,
    influences: impl Iterator<Item = (Scalar, ([Scalar; 4], [Scalar; 4]))>,
) -> Result<Vec3> {
    let mut qr = [0.; 4];
    let mut qd = [0.; 4];
    let mut reference = None;
    for (w, (q, d)) in influences {
        let (r, s) = *reference.get_or_insert((q, d));
        let dot: Scalar = (0..4).map(|i| q[i] * r[i] + d[i] * s[i]).sum();
        let sign = if dot >= 0. { 1. } else { -1. };
        for i in 0..4 {
            qr[i] += w * sign * q[i];
            qd[i] += w * sign * d[i];
        }
    }
    let norm = qr.iter().map(|v| v * v).sum::<Scalar>().sqrt();
    ensure(norm > 0., "degenerate dual quaternion blend")?;
    for i in 0..4 {
        qr[i] /= norm;
        qd[i] /= norm;
    }
    let conj = [-qr[0], -qr[1], -qr[2], qr[3]];
    let t = qmul(qd, conj);
    let r = qmul(qmul(qr, [v[0], v[1], v[2], 0.]), conj);
    Ok(Vec3::new(
        r[0] + 2. * t[0],
        r[1] + 2. * t[1],
        r[2] + 2. * t[2],
    ))
}
/// Weighted rigid registration; centered=false matches vector registration.
pub fn rigid_registration(
    source: &[Vec3],
    target: &[Vec3],
    weights: &[Scalar],
    centered: bool,
) -> Result<Mat4> {
    ensure(
        !source.is_empty() && source.len() == target.len() && target.len() == weights.len(),
        "registration size mismatch",
    )?;
    let sum: Scalar = weights.iter().sum();
    ensure(sum > 0. && sum.is_finite(), "invalid registration weights")?;
    let mut a = Vec3::zeros();
    let mut b = Vec3::zeros();
    if centered {
        for i in 0..source.len() {
            a += weights[i] * source[i] / sum;
            b += weights[i] * target[i] / sum;
        }
    }
    let mut h = Mat3::zeros();
    for i in 0..source.len() {
        h += weights[i] / sum * (target[i] - b) * (source[i] - a).transpose();
    }
    let r = special_procrustes(&h);
    Ok(rigid(&r, &(b - r * a)))
}
pub fn checked_rigid(h: &Mat4) -> Result<()> {
    ensure(
        h.iter().all(|x| x.is_finite())
            && (h[(3, 3)] - 1.).abs() < 1e-8
            && (0..3).all(|j| h[(3, j)].abs() < 1e-8),
        "pose must be finite affine 4x4",
    )
}
pub fn inverse(m: &Mat4) -> Result<Mat4> {
    m.try_inverse()
        .ok_or_else(|| Error::Invalid("singular matrix".into()))
}
