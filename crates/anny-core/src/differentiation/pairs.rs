//! First-order directional rules for Anny's vector, rotation and rigid math.
//! Values follow the reference kernels; derivatives use product/chain rules.
use crate::{ensure, math::*, Result};
#[derive(Clone, Copy)]
pub(super) struct V {
    pub v: Vec3,
    pub d: Vec3,
}
impl V {
    pub fn constant(v: Vec3) -> Self {
        Self {
            v,
            d: Vec3::zeros(),
        }
    }
    pub fn sub(self, b: Self) -> Self {
        Self {
            v: self.v - b.v,
            d: self.d - b.d,
        }
    }
    pub fn cross(self, b: Self) -> Self {
        Self {
            v: self.v.cross(&b.v),
            d: self.d.cross(&b.v) + self.v.cross(&b.d),
        }
    }
    pub fn norm(self) -> (f64, f64) {
        let n = self.v.norm();
        (n, if n == 0. { 0. } else { self.v.dot(&self.d) / n })
    }
    pub fn scaled(self, s: f64, ds: f64) -> Self {
        Self {
            v: self.v * s,
            d: self.d * s + self.v * ds,
        }
    }
    pub fn normalized(self, epsilon: f64) -> Self {
        let (n, dn) = self.norm();
        let denominator = n.max(epsilon);
        self.scaled(
            1. / denominator,
            if n > epsilon {
                -dn / (denominator * denominator)
            } else {
                0.
            },
        )
    }
}
#[derive(Clone, Copy)]
pub(super) struct M {
    pub v: Mat3,
    pub d: Mat3,
}
impl M {
    pub fn constant(v: Mat3) -> Self {
        Self {
            v,
            d: Mat3::zeros(),
        }
    }
    pub fn mul(self, b: Self) -> Self {
        Self {
            v: self.v * b.v,
            d: self.d * b.v + self.v * b.d,
        }
    }
    pub fn apply(self, b: V) -> V {
        V {
            v: self.v * b.v,
            d: self.d * b.v + self.v * b.d,
        }
    }
}
pub(super) fn outer(a: V, b: V) -> M {
    M {
        v: a.v * b.v.transpose(),
        d: a.d * b.v.transpose() + a.v * b.d.transpose(),
    }
}
pub(super) fn hat(v: Vec3) -> Mat3 {
    Mat3::new(0., -v[2], v[1], v[2], 0., -v[0], -v[1], v[0], 0.)
}
pub(super) fn vee(m: Mat3) -> Vec3 {
    Vec3::new(m[(2, 1)], m[(0, 2)], m[(1, 0)])
}
pub(super) fn exponential(a: V) -> M {
    let t = a.v.norm();
    let value = rotvec(&a.v);
    if t < 1e-6 {
        return M {
            v: value,
            d: hat(a.d),
        };
    }
    let h = hat(a.v);
    let t2 = t * t;
    let (c1, c2) = if t < 1e-3 {
        (
            0.5 - t2 / 24. + t2 * t2 / 720.,
            1. / 6. - t2 / 120. + t2 * t2 / 5040.,
        )
    } else {
        ((1. - t.cos()) / t2, (t - t.sin()) / (t2 * t))
    };
    let right = Mat3::identity() - h * c1 + h * h * c2;
    M {
        v: value,
        d: value * hat(right * a.d),
    }
}
/// Differentiate the closest proper rotation without differentiating SVD vectors.
/// R^T A is symmetric; its Sylvester equation gives R^T dR. Non-unique
/// rotations with a nonzero perturbation return an error, not fabricated zeros.
pub(super) fn procrustes(a: M) -> Result<M> {
    let r = special_procrustes(&a.v);
    if a.d.norm_squared() == 0. {
        return Ok(M::constant(r));
    }
    let s = (r.transpose() * a.v + a.v.transpose() * r) * 0.5;
    let k = Mat3::identity() * s.trace() - s;
    let scale = k.norm();
    ensure(
        scale > 0. && scale.is_finite(),
        "non-differentiable zero orientation covariance",
    )?;
    let normalized = k / scale;
    let svd = normalized.svd(false, false);
    ensure(
        svd.singular_values[2] > 1e-10,
        "orientation derivative is non-unique or ill-conditioned",
    )?;
    let inverse = normalized
        .try_inverse()
        .ok_or_else(|| crate::Error::Invalid("singular orientation derivative".into()))?;
    let b = vee(r.transpose() * a.d - a.d.transpose() * r) / scale;
    let d = r * hat(inverse * b);
    ensure(
        d.iter().all(|x| x.is_finite()),
        "non-finite orientation derivative",
    )?;
    Ok(M { v: r, d })
}
pub(super) fn tail(head: V, tail: V, roll: Mat3) -> H {
    let y = tail.sub(head).normalized(0.);
    let cross = y.cross(V::constant(Vec3::y()));
    let (n, dn) = cross.norm();
    let axis = cross.scaled(1. / n, -dn / (n * n));
    let angle = n.atan2(y.v[1]);
    let da = (y.v[1] * dn - n * y.d[1]) / (n * n + y.v[1] * y.v[1]);
    let rotation = if (axis.v.norm_squared() - 1.).abs() < 0.1 {
        exponential(axis.scaled(-angle, -da))
    } else {
        M::constant(Mat3::from_diagonal(&Vec3::new(1., -1., -1.)))
    };
    H::rigid(rotation.mul(M::constant(roll)), head)
}
pub(super) fn shortest(target: V, source: V) -> M {
    let a = target.normalized(1e-8);
    let b = source.normalized(1e-8);
    let raw = a.v.dot(&b.v);
    let dot = raw.clamp(-1., 1.);
    let ddot = if (-1. ..=1.).contains(&raw) {
        a.d.dot(&b.v) + a.v.dot(&b.d)
    } else {
        0.
    };
    if dot < -1. + 1e-6 {
        let axis = b
            .cross(V::constant(if b.v[0].abs() > 0.6 {
                Vec3::y()
            } else {
                Vec3::x()
            }))
            .normalized(0.);
        let m = outer(axis, axis);
        return M {
            v: m.v * 2. - Mat3::identity(),
            d: m.d * 2.,
        };
    }
    let v = b.cross(a);
    let h = hat(v.v);
    let dh = hat(v.d);
    let den = 1. + dot;
    M {
        v: Mat3::identity() + h + h * h / den,
        d: dh + (dh * h + h * dh) / den - h * h * (ddot / (den * den)),
    }
}
#[derive(Clone, Copy)]
pub(super) struct H {
    pub v: Mat4,
    pub d: Mat4,
}
impl H {
    pub fn constant(v: Mat4) -> Self {
        Self {
            v,
            d: Mat4::zeros(),
        }
    }
    pub fn rotation(self) -> M {
        M {
            v: rotation(&self.v),
            d: rotation(&self.d),
        }
    }
    pub fn translation(self) -> V {
        V {
            v: translation(&self.v),
            d: translation(&self.d),
        }
    }
    pub fn rigid(r: M, t: V) -> Self {
        let mut d = Mat4::zeros();
        d.fixed_view_mut::<3, 3>(0, 0).copy_from(&r.d);
        d.fixed_view_mut::<3, 1>(0, 3).copy_from(&t.d);
        Self {
            v: rigid(&r.v, &t.v),
            d,
        }
    }
    pub fn mul(self, b: Self) -> Self {
        Self {
            v: self.v * b.v,
            d: self.d * b.v + self.v * b.d,
        }
    }
    pub fn inverse(self) -> Self {
        let r = self.rotation();
        let t = self.translation();
        let rt = M {
            v: r.v.transpose(),
            d: r.d.transpose(),
        };
        let tr = rt.apply(t).scaled(-1., 0.);
        Self::rigid(rt, tr)
    }
    pub fn apply(self, b: V) -> V {
        let p = self.rotation().apply(b);
        let t = self.translation();
        V {
            v: p.v + t.v,
            d: p.d + t.d,
        }
    }
}
#[derive(Clone, Copy)]
pub(super) struct Q {
    pub v: [f64; 4],
    pub d: [f64; 4],
}
impl Q {
    fn constant(v: [f64; 4]) -> Self {
        Self { v, d: [0.; 4] }
    }
    fn mul(self, b: Self) -> Self {
        let x = qmul(self.d, b.v);
        let y = qmul(self.v, b.d);
        Self {
            v: qmul(self.v, b.v),
            d: std::array::from_fn(|i| x[i] + y[i]),
        }
    }
    fn conjugate(self) -> Self {
        Self {
            v: [-self.v[0], -self.v[1], -self.v[2], self.v[3]],
            d: [-self.d[0], -self.d[1], -self.d[2], self.d[3]],
        }
    }
}
pub(super) fn dual(h: H) -> (Q, Q) {
    let r = h.rotation();
    let value = quaternion(&r.v);
    let w = vee((r.v.transpose() * r.d - r.d.transpose() * r.v) * 0.5);
    let tangent = qmul(value, [w[0] * 0.5, w[1] * 0.5, w[2] * 0.5, 0.]);
    let real = Q {
        v: value,
        d: tangent,
    };
    let t = h.translation();
    let translation = Q {
        v: [t.v[0] * 0.5, t.v[1] * 0.5, t.v[2] * 0.5, 1.],
        d: [t.d[0] * 0.5, t.d[1] * 0.5, t.d[2] * 0.5, 0.],
    };
    (real, translation.mul(real))
}
pub(super) fn dqs(p: V, influences: impl Iterator<Item = (f64, (Q, Q))>) -> Result<V> {
    let mut qr = Q::constant([0.; 4]);
    let mut qd = qr;
    let mut reference = None;
    for (weight, (r, d)) in influences {
        let (a, b) = *reference.get_or_insert((r, d));
        let dot = (0..4)
            .map(|i| r.v[i] * a.v[i] + d.v[i] * b.v[i])
            .sum::<f64>();
        let sign = if dot >= 0. { 1. } else { -1. };
        for i in 0..4 {
            qr.v[i] += weight * sign * r.v[i];
            qr.d[i] += weight * sign * r.d[i];
            qd.v[i] += weight * sign * d.v[i];
            qd.d[i] += weight * sign * d.d[i];
        }
    }
    let norm = qr.v.iter().map(|v| v * v).sum::<f64>().sqrt();
    ensure(norm > 0., "degenerate differential DQS blend")?;
    let dn = (0..4).map(|i| qr.v[i] * qr.d[i]).sum::<f64>() / norm;
    for i in 0..4 {
        qr.d[i] = qr.d[i] / norm - qr.v[i] * dn / (norm * norm);
        qd.d[i] = qd.d[i] / norm - qd.v[i] * dn / (norm * norm);
        qr.v[i] /= norm;
        qd.v[i] /= norm;
    }
    let conjugate = qr.conjugate();
    let translation = qd.mul(conjugate);
    let rotated = qr
        .mul(Q {
            v: [p.v[0], p.v[1], p.v[2], 0.],
            d: [p.d[0], p.d[1], p.d[2], 0.],
        })
        .mul(conjugate);
    Ok(V {
        v: Vec3::new(
            rotated.v[0] + 2. * translation.v[0],
            rotated.v[1] + 2. * translation.v[1],
            rotated.v[2] + 2. * translation.v[2],
        ),
        d: Vec3::new(
            rotated.d[0] + 2. * translation.d[0],
            rotated.d[1] + 2. * translation.d[1],
            rotated.d[2] + 2. * translation.d[2],
        ),
    })
}
