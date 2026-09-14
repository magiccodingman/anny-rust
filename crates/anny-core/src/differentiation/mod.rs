//! Specialized first-order differentiation of Anny's continuous parameters.
//! No interpreter, finite-difference perturbation, or general ML framework.
//! Directions use named scalar controls and left-trivialized bone rotation
//! increments (radians), with independent additive translations (meters).
mod pairs;
mod reverse;
pub use reverse::{vjp, ParameterSelection};

use crate::{
    config::{BoneOrientation, PHENOTYPE_VARIATIONS},
    ensure,
    math::*,
    model, Anny, ModelOutput, Parameters, PoseParameterization, Result, SkinningMethod, Tensor,
};
use pairs::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ParameterDirection {
    pub phenotypes: BTreeMap<String, f64>,
    pub local_changes: BTreeMap<String, f64>,
    pub facial_actions: BTreeMap<String, f64>,
    pub bone_rotations: BTreeMap<String, [f64; 3]>,
    pub bone_translations: BTreeMap<String, [f64; 3]>,
}
fn validate_names<T>(values: &BTreeMap<String, T>, labels: &[String]) -> Result<()> {
    for name in values.keys() {
        ensure(
            labels.contains(name),
            format!("unknown differentiated parameter {name}"),
        )?;
    }
    Ok(())
}
fn coefficient_direction(model: &Anny, p: &Parameters, dir: &ParameterDirection) -> Result<Tensor> {
    validate_names(&dir.phenotypes, &model.phenotype_labels)?;
    validate_names(&dir.local_changes, &model.local_change_labels)?;
    validate_names(&dir.facial_actions, &model.facial_action_labels)?;
    ensure(
        dir.phenotypes
            .values()
            .chain(dir.local_changes.values())
            .chain(dir.facial_actions.values())
            .all(|v| v.is_finite()),
        "non-finite parameter direction",
    )?;
    let ph = model::parse_values(
        &p.phenotype_kwargs,
        &model.phenotype_labels,
        0.5,
        "phenotypes",
    )?;
    let lo = model::parse_values(
        &p.local_changes_kwargs,
        &model.local_change_labels,
        0.,
        "local_changes",
    )?;
    let fa = model::parse_values(
        &p.facial_actions,
        &model.facial_action_labels,
        0.,
        "facial_actions",
    )?;
    ensure(
        ph.shape[0] == 1 && lo.shape[0] == 1 && fa.shape[0] == 1,
        "differentiation currently accepts one character per call",
    )?;
    let value = |key: &str| {
        model
            .phenotype_labels
            .iter()
            .position(|n| n == key)
            .map_or(0.5, |i| ph.data[i])
    };
    let derivative = |key: &str| *dir.phenotypes.get(key).unwrap_or(&0.);
    let race = [value("african"), value("asian"), value("caucasian")];
    let drace = [
        derivative("african"),
        derivative("asian"),
        derivative("caucasian"),
    ];
    let sum: f64 = race.iter().sum();
    let dsum: f64 = drace.iter().sum();
    let mut features = Vec::with_capacity(26);
    for (name, anchors_names) in PHENOTYPE_VARIATIONS {
        if *name == "race" {
            for i in 0..3 {
                let v = race[i] / sum;
                features.push(if v.is_finite() {
                    (v, (drace[i] - v * dsum) / sum)
                } else {
                    (1. / 3., 0.)
                });
            }
            continue;
        }
        let min = if *name == "age" { -1. / 3. } else { 0. };
        let anchors = (0..anchors_names.len())
            .map(|i| min + (1. - min) * i as f64 / (anchors_names.len() - 1) as f64)
            .collect::<Vec<_>>();
        let x = value(name);
        let w = linear_interpolation(x, &anchors, model.config.extrapolate_phenotypes)?;
        let i = anchors
            .partition_point(|&a| a < x)
            .clamp(1, anchors.len() - 1);
        let dx = if model.config.extrapolate_phenotypes
            || (x >= anchors[0] && x <= anchors[anchors.len() - 1])
        {
            derivative(name) / (anchors[i] - anchors[i - 1])
        } else {
            0.
        };
        for (k, v) in w.into_iter().enumerate() {
            features.push((
                v,
                if k == i {
                    dx
                } else if k + 1 == i {
                    -dx
                } else {
                    0.
                },
            ));
        }
    }
    let mask = model.data.get("stacked_phenotype_blend_shapes_mask")?;
    let mut out = Tensor::zeros(vec![1, model.data.blendshape_count()]);
    for i in 0..mask.shape[0] {
        let (mut v, mut d) = (1., 0.);
        for (j, &(x, dx)) in features.iter().enumerate() {
            if mask.data[i * 26 + j] != 0. {
                d = d * x + v * dx;
                v *= x;
            }
        }
        out.data[i] = d;
    }
    let off = mask.shape[0];
    for (i, name) in model.facial_action_labels.iter().enumerate() {
        out.data[off + i] = *dir.facial_actions.get(name).unwrap_or(&0.);
    }
    let off = off + fa.shape[1];
    for (i, name) in model.local_change_labels.iter().enumerate() {
        let d = *dir.local_changes.get(name).unwrap_or(&0.);
        // Upstream deliberately differentiates both masked branches at zero.
        // A ReLU-style zero derivative would leave newly enabled locals dead.
        out.data[off + 2 * i] = if lo.data[i] >= 0. { d } else { 0. };
        out.data[off + 2 * i + 1] = if lo.data[i] <= 0. { -d } else { 0. };
    }
    out.validate()?;
    Ok(out)
}
fn blend_direction(m: &Anny, name: &str, coefficients: &Tensor) -> Result<Tensor> {
    let basis = m.data.get(name)?;
    let shape = basis.shape[1..].to_vec();
    let width: usize = shape.iter().product();
    let mut out = Tensor::zeros(std::iter::once(1).chain(shape).collect::<Vec<_>>());
    for (i, &coefficient) in coefficients.data.iter().enumerate() {
        if coefficient != 0. {
            for (j, v) in out.data.iter_mut().enumerate() {
                *v += basis.data[i * width + j] * coefficient;
            }
        }
    }
    out.validate()?;
    Ok(out)
}
fn vectors(value: &Tensor, derivative: &Tensor) -> Vec<V> {
    value
        .data
        .chunks_exact(3)
        .zip(derivative.data.chunks_exact(3))
        .map(|(v, d)| V {
            v: vec3(v),
            d: vec3(d),
        })
        .collect()
}
type RestDifferential = (Vec<V>, Vec<H>, Option<Vec<V>>);
fn rest_differential(
    m: &Anny,
    coefficients: &Tensor,
    dc: &Tensor,
    base: &ModelOutput,
) -> Result<RestDifferential> {
    let rig = m.resolved_rig();
    let j = m.data.bone_count();
    let vertices = vectors(
        base.get("rest_vertices")?,
        &blend_direction(m, "blendshapes", dc)?,
    );
    let heads = vectors(
        base.get("rest_bone_heads")?,
        &blend_direction(m, "bone_heads_blendshapes", dc)?,
    );
    let mut tails = None;
    let mut orientations = vec![M::constant(Mat3::identity()); j];
    match rig.bone_orientation {
        BoneOrientation::Blender => {
            let values = vectors(
                base.get("rest_bone_tails")?,
                &blend_direction(m, "bone_tails_blendshapes", dc)?,
            );
            let rolls = m.data.get("bone_rolls_rotmat")?;
            for i in 0..j {
                orientations[i] =
                    tail(heads[i], values[i], mat3(&rolls.data[i * 9..i * 9 + 9])).rotation();
            }
            tails = Some(values);
        }
        BoneOrientation::Cached => {
            let a = model::apply_blendshapes(
                m.data.get("bone_template_orientation_matrices")?,
                m.data.get("bone_orientation_blendshapes")?,
                coefficients,
            )?;
            let da = blend_direction(m, "bone_orientation_blendshapes", dc)?;
            for (i, orientation) in orientations.iter_mut().enumerate() {
                *orientation = procrustes(M {
                    v: mat3(&a.data[i * 9..i * 9 + 9]),
                    d: mat3(&da.data[i * 9..i * 9 + 9]),
                })?;
            }
            if m.data.arrays.contains_key("bone_children_indices") {
                refine(m, &heads, &mut orientations)?;
            }
        }
        BoneOrientation::Procrustes => {
            let mask = m.data.get("bone_nonzeroweight_mask")?;
            let ids = m.data.get("bone_vertex_indices")?;
            let weights = m.data.get("bone_vertex_weights")?;
            let source = m.data.get("template_bone_vertices")?;
            let mut row = 0;
            let width = ids.shape[1];
            for i in 0..j {
                if mask.data[i] == 0. {
                    continue;
                }
                let mut a = M::constant(Mat3::zeros());
                for k in 0..width {
                    let w = weights.data[row * width + k];
                    if w == 0. {
                        continue;
                    }
                    let id = ids.data[row * width + k] as usize;
                    let item = outer(
                        vertices[id].sub(heads[i]),
                        V::constant(vec3(
                            &source.data[(row * width + k) * 3..(row * width + k + 1) * 3],
                        )),
                    );
                    a.v += item.v * w;
                    a.d += item.d * w;
                }
                orientations[i] = procrustes(a)?;
                row += 1;
            }
        }
    }
    if rig.root_identity_orientation {
        orientations[0] = M::constant(Mat3::identity());
    }
    let reference = base.get("rest_bone_poses")?;
    let poses = orientations
        .iter()
        .zip(&heads)
        .enumerate()
        .map(|(i, (&r, &t))| {
            let mut h = H::rigid(r, t);
            h.v = mat4(&reference.data[i * 16..i * 16 + 16]);
            h
        })
        .collect();
    Ok((vertices, poses, tails))
}
fn refine(m: &Anny, heads: &[V], orientations: &mut [M]) -> Result<()> {
    let idx = m.data.get("bone_children_indices")?;
    let mask = m.data.get("bone_children_mask")?;
    let off = m.data.get("bone_children_local_offsets")?;
    let j = m.data.bone_count();
    let k = idx.shape[1];
    let original = orientations.to_vec();
    for bone in 1..j {
        let children = (0..k)
            .filter(|&i| mask.data[bone * k + i] != 0.)
            .collect::<Vec<_>>();
        if children.is_empty() {
            continue;
        }
        let mut a = vec![];
        let mut b = vec![];
        for &i in &children {
            let id = idx.data[bone * k + i] as usize;
            a.push(heads[id].sub(heads[bone]));
            b.push(original[bone].apply(V::constant(vec3(
                &off.data[(bone * k + i) * 3..(bone * k + i + 1) * 3],
            ))));
        }
        let alignment = if a.len() == 1 {
            shortest(a[0], b[0])
        } else {
            let mut h = M::constant(Mat3::zeros());
            for (&a, &b) in a.iter().zip(&b) {
                let item = outer(a, b);
                h.v += item.v;
                h.d += item.d;
            }
            let na = a[0].cross(a[1]);
            let nb = b[0].cross(b[1]);
            let (an, dan) = na.norm();
            let (bn, dbn) = nb.norm();
            if an > 1e-9 && bn > 1e-9 {
                let (av, dav) = a[0].norm();
                let (bv, dbv) = b[0].norm();
                let va = na.scaled(
                    av / (an + 1e-8),
                    dav / (an + 1e-8) - av * dan / (an + 1e-8).powi(2),
                );
                let vb = nb.scaled(
                    bv / (bn + 1e-8),
                    dbv / (bn + 1e-8) - bv * dbn / (bn + 1e-8).powi(2),
                );
                let item = outer(va, vb);
                h.v += item.v;
                h.d += item.d;
            }
            procrustes(h)?
        };
        orientations[bone] = alignment.mul(original[bone]);
    }
    let before = orientations.to_vec();
    for bone in 1..j {
        if !m.data.metadata.bone_parents.contains(&(bone as i32)) {
            orientations[bone] = before[m.data.metadata.bone_parents[bone] as usize];
        }
    }
    Ok(())
}
fn forward(parents: &[i32], rest: &[H], delta: &[H], base: Option<H>) -> Result<Vec<H>> {
    let mut poses = vec![H::constant(Mat4::identity()); rest.len()];
    let mut transforms = poses.clone();
    for i in propagation_order(parents)? {
        let item = rest[i].mul(delta[i]);
        poses[i] = if parents[i] < 0 {
            base.map_or(item, |b| b.mul(item))
        } else {
            transforms[parents[i] as usize].mul(item)
        };
        transforms[i] = poses[i].mul(rest[i].inverse());
    }
    Ok(poses)
}
fn absolute(
    parents: &[i32],
    rest: &[H],
    rotations: &[M],
    base: Option<H>,
) -> Result<(Vec<H>, Vec<H>)> {
    let mut poses = vec![H::constant(Mat4::identity()); rest.len()];
    let mut transforms = poses.clone();
    for i in propagation_order(parents)? {
        let item = if parents[i] < 0 {
            base.map_or(rest[i], |b| b.mul(rest[i]))
        } else {
            transforms[parents[i] as usize].mul(rest[i])
        };
        poses[i] = H::rigid(rotations[i], item.translation());
        transforms[i] = poses[i].mul(rest[i].inverse());
    }
    Ok((poses, transforms))
}
fn posed(
    m: &Anny,
    rest: &[H],
    delta: &[H],
    mode: PoseParameterization,
) -> Result<(Vec<H>, Vec<H>)> {
    let parents = &m.data.metadata.bone_parents;
    let poses = match mode {
        PoseParameterization::World => delta.to_vec(),
        PoseParameterization::WorldOrient => {
            return absolute(
                parents,
                rest,
                &delta.iter().map(|h| h.rotation()).collect::<Vec<_>>(),
                Some(delta[0].mul(rest[0].inverse())),
            );
        }
        _ => {
            let refs = if let Some(r) = m.data.arrays.get("reference_bone_orientations") {
                absolute(
                    parents,
                    rest,
                    &r.data
                        .chunks_exact(9)
                        .map(|x| M::constant(mat3(x)))
                        .collect::<Vec<_>>(),
                    None,
                )?
                .0
            } else {
                rest.to_vec()
            };
            let base = if mode == PoseParameterization::LocalBoneWorld {
                None
            } else {
                Some(refs[0].inverse())
            };
            let deltas = delta
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    if mode == PoseParameterization::LocalRef {
                        let r = H::rigid(refs[i].rotation(), V::constant(Vec3::zeros()));
                        r.inverse().mul(*t).mul(r)
                    } else {
                        *t
                    }
                })
                .collect::<Vec<_>>();
            forward(parents, &refs, &deltas, base)?
        }
    };
    let transforms = poses
        .iter()
        .zip(rest)
        .map(|(&p, &r)| p.mul(r.inverse()))
        .collect();
    Ok((poses, transforms))
}
/// Return the Jacobian-vector product for a single character. Branch decisions
/// follow the evaluated point. At non-unique orientation projections the function
/// returns an error rather than claiming a derivative. No epsilon is used.
/// At a zero local morph the upstream masked-branch subgradient is used: both
/// signed branches contribute. This is not a classical derivative at the kink.
pub fn jvp(m: &Anny, p: &Parameters, direction: &ParameterDirection) -> Result<ModelOutput> {
    let base = m.forward(p)?;
    jvp_at(m, p, direction, &base)
}
pub(super) fn jvp_at(
    m: &Anny,
    p: &Parameters,
    direction: &ParameterDirection,
    base: &ModelOutput,
) -> Result<ModelOutput> {
    ensure(
        base.get("vertices")?.shape[0] == 1,
        "differentiation accepts one character per call",
    )?;
    validate_names(&direction.bone_rotations, &m.data.metadata.bone_labels)?;
    validate_names(&direction.bone_translations, &m.data.metadata.bone_labels)?;
    ensure(
        direction
            .bone_rotations
            .values()
            .chain(direction.bone_translations.values())
            .flatten()
            .all(|v| v.is_finite()),
        "non-finite pose direction",
    )?;
    let c = m.coefficients(p)?;
    let dc = coefficient_direction(m, p, direction)?;
    let (rest, restposes, tails) = rest_differential(m, &c, &dc, base)?;
    let input = model::parse_pose(&p.pose_parameters, &m.data.metadata.bone_labels)?;
    ensure(
        input.shape[0] == 1,
        "differentiation pose cannot be a batch",
    )?;
    let mut delta = Vec::new();
    for (i, row) in input.data.chunks_exact(16).enumerate() {
        let v = mat4(row);
        let r = rotation(&v);
        ensure(
            (r.transpose() * r - Mat3::identity()).norm() < 1e-4
                && (r.determinant() - 1.).abs() < 1e-4,
            "rotation derivatives require proper rigid pose inputs",
        )?;
        let name = &m.data.metadata.bone_labels[i];
        let angular = direction
            .bone_rotations
            .get(name)
            .copied()
            .unwrap_or([0.; 3]);
        let tr = direction
            .bone_translations
            .get(name)
            .copied()
            .unwrap_or([0.; 3]);
        delta.push(H::rigid(
            M {
                v: r,
                d: hat(vec3(&angular)) * r,
            },
            V {
                v: translation(&v),
                d: vec3(&tr),
            },
        ));
    }
    let (poses, transforms) = posed(
        m,
        &restposes,
        &delta,
        p.pose_parameterization
            .unwrap_or(m.config.pose_parameterization),
    )?;
    let weights = m.data.get("vertex_bone_weights")?;
    let ids = m.data.get("vertex_bone_indices")?;
    let width = weights.shape[1];
    let dq = if m.config.skinning_method == SkinningMethod::Dqs {
        transforms.iter().copied().map(dual).collect::<Vec<_>>()
    } else {
        vec![]
    };
    let mut vertices = Tensor::zeros(vec![1, rest.len(), 3]);
    for (i, &v) in rest.iter().enumerate() {
        let d = if m.config.skinning_method == SkinningMethod::Dqs {
            dqs(
                v,
                (0..width).map(|k| {
                    let slot = i * width + k;
                    (weights.data[slot], dq[ids.data[slot] as usize])
                }),
            )?
            .d
        } else {
            let mut out = Vec3::zeros();
            for k in 0..width {
                let slot = i * width + k;
                out += transforms[ids.data[slot] as usize].apply(v).d * weights.data[slot];
            }
            out
        };
        vertices.data[i * 3..i * 3 + 3].copy_from_slice(d.as_slice());
    }
    let mut arrays = BTreeMap::from([
        ("vertices".into(), vertices),
        ("rest_vertices".into(), derivative_vectors(&rest)),
        (
            "rest_bone_heads".into(),
            derivative_vectors(
                &restposes
                    .iter()
                    .map(|h| h.translation())
                    .collect::<Vec<_>>(),
            ),
        ),
        ("bone_poses".into(), derivative_matrices(&poses)),
        ("rest_bone_poses".into(), derivative_matrices(&restposes)),
    ]);
    if let Some(tails) = tails {
        arrays.insert("rest_bone_tails".into(), derivative_vectors(&tails));
        if p.return_bone_ends {
            arrays.insert(
                "bone_heads".into(),
                derivative_vectors(
                    &restposes
                        .iter()
                        .zip(&transforms)
                        .map(|(r, t)| t.apply(r.translation()))
                        .collect::<Vec<_>>(),
                ),
            );
            arrays.insert(
                "bone_tails".into(),
                derivative_vectors(
                    &tails
                        .iter()
                        .zip(&transforms)
                        .map(|(&v, &t)| t.apply(v))
                        .collect::<Vec<_>>(),
                ),
            );
        }
    }
    for tensor in arrays.values() {
        tensor.validate()?;
    }
    Ok(ModelOutput { arrays })
}
fn derivative_vectors(v: &[V]) -> Tensor {
    Tensor {
        shape: vec![1, v.len(), 3],
        data: v.iter().flat_map(|x| [x.d[0], x.d[1], x.d[2]]).collect(),
        kind: crate::tensor::Kind::Float,
    }
}
fn derivative_matrices(v: &[H]) -> Tensor {
    let mut t = Tensor::zeros(vec![1, v.len(), 4, 4]);
    for (h, out) in v.iter().zip(t.data.chunks_exact_mut(16)) {
        write4(&h.d, out);
    }
    t
}
