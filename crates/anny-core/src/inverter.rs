// Port of NAVER Anny's finite-difference / joint-registration fitting algorithm.
// Copyright (C) 2025 NAVER Corp. SPDX-License-Identifier: Apache-2.0
use crate::{
    ensure,
    math::*,
    model::{self, Anny, ModelOutput, Parameters},
    Error, PoseParameterization, Result, Tensor,
};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct InverterOptions {
    pub eps: f64,
    pub n_points: Option<usize>,
    pub max_n_iters: usize,
    pub joint_min_weight: f64,
    pub joint_top_k: Option<usize>,
    pub identity_bone_labels: Vec<String>,
    pub regularization: BTreeMap<String, f64>,
    /// Explicit calibration for an optional post_gd shape prior.
    pub shape_prior: Option<crate::distribution::SimpleShapeDistribution>,
}
impl Default for InverterOptions {
    fn default() -> Self {
        Self {
            eps: 0.1,
            n_points: None,
            max_n_iters: 10,
            joint_min_weight: 0.01,
            joint_top_k: Some(1024),
            identity_bone_labels: vec![],
            regularization: BTreeMap::new(),
            shape_prior: None,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FitOptions {
    pub initial_phenotype_kwargs: Value,
    pub initial_pose_parameters: Value,
    pub optimize_phenotypes: bool,
    pub excluded_phenotypes: Vec<String>,
    pub max_n_iters: Option<usize>,
    pub max_delta: f64,
    pub shared_phenotypes: bool,
    pub multistart: BTreeMap<String, Vec<f64>>,
    pub post_gd: bool,
    pub post_gd_steps: usize,
    pub post_gd_lr: f64,
    pub post_gd_prior_weight: f64,
    pub post_gd_optimize_local_changes: bool,
    pub post_gd_optimize_facial_actions: bool,
}
impl Default for FitOptions {
    fn default() -> Self {
        Self {
            initial_phenotype_kwargs: Value::Null,
            initial_pose_parameters: Value::Null,
            optimize_phenotypes: true,
            excluded_phenotypes: vec![],
            max_n_iters: None,
            max_delta: 0.1,
            shared_phenotypes: false,
            multistart: BTreeMap::new(),
            post_gd: false,
            post_gd_steps: 100,
            post_gd_lr: 1e-3,
            post_gd_prior_weight: 0.,
            post_gd_optimize_local_changes: false,
            post_gd_optimize_facial_actions: false,
        }
    }
}
#[derive(Clone, Debug)]
pub struct FitResult {
    pub parameters: Parameters,
    pub vertices: Tensor,
    pub output: ModelOutput,
    pub mean_vertex_error: Vec<f64>,
    pub iterations: usize,
    /// Empty when refinement was disabled; initial loss plus each Adam step otherwise.
    pub post_gd_losses: Vec<f64>,
}
impl FitResult {
    pub fn to_json(&self) -> Value {
        json!({"parameters":self.parameters,"mean_vertex_error":self.mean_vertex_error,"iterations":self.iterations,"post_gd_losses":self.post_gd_losses})
    }
}
/// The baseline uses finite differences/registration; optional post_gd uses native
/// analytic parameter gradients. All matrices and
/// regularized solves are f64; iteration trajectories need not match torch f32.
pub struct AnnyInverter<'a> {
    model: &'a Anny,
    options: InverterOptions,
    unique: Vec<usize>,
    points: Vec<usize>,
    partition: Vec<Vec<(usize, f64)>>,
    identity: BTreeSet<usize>,
    max_weight: f64,
}
#[derive(Clone)]
struct State {
    pose: Tensor,
    phenotypes: Tensor,
}
impl<'a> AnnyInverter<'a> {
    pub fn new(model: &'a Anny, options: InverterOptions) -> Result<Self> {
        ensure(
            options.eps > 0. && options.eps.is_finite() && options.joint_min_weight >= 0.,
            "invalid inverter epsilon/weight threshold",
        )?;
        ensure(
            options.joint_top_k != Some(0) && options.n_points != Some(0),
            "inverter sample counts must be positive",
        )?;
        for (k, v) in &options.regularization {
            ensure(
                model.phenotype_labels.contains(k) && v.is_finite() && *v >= 0.,
                "invalid phenotype regularization",
            )?;
        }
        let unique = model
            .data
            .get("faces")?
            .checked_indices(model.data.vertex_count(), "inverter faces")?
            .into_iter()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        ensure(!unique.is_empty(), "fitting requires a nonempty surface")?;
        let points = if let Some(count) = options.n_points {
            if count == 1 {
                vec![0]
            } else {
                (0..count)
                    .map(|i| i * (unique.len() - 1) / (count - 1))
                    .collect()
            }
        } else {
            (0..unique.len()).collect()
        };
        let j = model.data.bone_count();
        let w = model.data.get("vertex_bone_weights")?;
        let ids = model.data.get("vertex_bone_indices")?;
        let k = w.shape[1];
        let mut partition = vec![Vec::new(); j];
        for (i, &v) in unique.iter().enumerate() {
            for s in 0..k {
                let weight = w.data[v * k + s];
                if weight >= options.joint_min_weight && weight > 0. {
                    partition[ids.data[v * k + s] as usize].push((i, weight));
                }
            }
        }
        for entries in &mut partition {
            if let Some(count) = options.joint_top_k {
                if entries.len() > count {
                    entries.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                    entries.truncate(count);
                }
            }
            let sum = entries.iter().map(|x| x.1).sum::<f64>() + 1e-8;
            for (_, w) in entries {
                *w /= sum;
            }
        }
        let max_weight = partition.iter().flatten().map(|x| x.1).fold(0., f64::max);
        let identity = options
            .identity_bone_labels
            .iter()
            .map(|name| {
                model
                    .data
                    .metadata
                    .bone_labels
                    .iter()
                    .position(|s| s == name)
                    .ok_or_else(|| Error::Invalid(format!("unknown identity bone {name}")))
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            model,
            options,
            unique,
            points,
            partition,
            identity,
            max_weight,
        })
    }
    fn parameters(&self, state: &State) -> Parameters {
        Parameters {
            phenotype_kwargs: state.phenotypes.nested_json(),
            pose_parameters: state.pose.nested_json(),
            pose_parameterization: Some(PoseParameterization::LocalBone),
            ..Default::default()
        }
    }
    fn evaluate(&self, state: &State) -> Result<ModelOutput> {
        self.model.forward(&self.parameters(state))
    }
    fn vertices(&self, out: &ModelOutput) -> Result<Tensor> {
        out.get("vertices")?.select(1, &self.unique)
    }
    fn initial(&self, options: &FitOptions, b: usize) -> Result<State> {
        let mut initial = options.initial_phenotype_kwargs.clone();
        if initial.is_null() {
            initial = json!({});
        }
        if let Some(m) = initial.as_object_mut() {
            if !m.contains_key("age") {
                m.insert("age".into(), json!(0.8));
            }
        }
        let ph = model::parse_values(
            &initial,
            &self.model.phenotype_labels,
            0.5,
            "initial_phenotype_kwargs",
        )?;
        let pose = model::parse_pose(
            &options.initial_pose_parameters,
            &self.model.data.metadata.bone_labels,
        )?;
        model::broadcast(&[b, ph.shape[0], pose.shape[0]])?;
        Ok(State {
            pose: repeat_batch(&pose, b)?,
            phenotypes: repeat_batch(&ph, b)?,
        })
    }
    fn jointwise(
        &self,
        reference: &ModelOutput,
        target: &Tensor,
        state: &State,
    ) -> Result<(Tensor, Tensor)> {
        let vref = self.vertices(reference)?;
        let bones = reference.get("bone_poses")?;
        let b = target.shape[0];
        let j = self.model.data.bone_count();
        let n = self.unique.len();
        let mut absolute = model::identity_poses(b, j);
        for bi in 0..b {
            for bone in 0..j {
                let entries = &self.partition[bone];
                let index = (bi * j + bone) * 16;
                let mut transform = Mat4::identity();
                if !entries.is_empty() {
                    let mut a = Vec::with_capacity(entries.len() + 1);
                    let mut z = Vec::with_capacity(entries.len() + 1);
                    let mut weights = Vec::with_capacity(entries.len() + 1);
                    let mut ca = Vec3::zeros();
                    let mut cz = Vec3::zeros();
                    let mut sum = 0.;
                    for &(v, w) in entries {
                        let va = vec3(&vref.data[(bi * n + v) * 3..(bi * n + v + 1) * 3]);
                        let vz = vec3(&target.data[(bi * n + v) * 3..(bi * n + v + 1) * 3]);
                        a.push(va);
                        z.push(vz);
                        weights.push(w);
                        ca += w * va;
                        cz += w * vz;
                        sum += w;
                    }
                    a.push(ca / (sum + 1e-8));
                    z.push(cz / (sum + 1e-8));
                    weights.push(2. * self.max_weight);
                    transform = rigid_registration(&a, &z, &weights, true)?;
                }
                write4(
                    &(transform * mat4(&bones.data[index..index + 16])),
                    &mut absolute.data[index..index + 16],
                );
            }
        }
        let output_abs = self.model.forward(&Parameters {
            phenotype_kwargs: state.phenotypes.nested_json(),
            pose_parameters: absolute.nested_json(),
            pose_parameterization: Some(PoseParameterization::World),
            ..Default::default()
        })?;
        let mut pose = self
            .model
            .pose_parameters(&output_abs, PoseParameterization::LocalBone)?;
        for bi in 0..b {
            for bone in 0..j {
                let index = (bi * j + bone) * 16;
                let h = mat4(&pose.data[index..index + 16]);
                let r = if bone == 0
                    || self.partition[bone].is_empty()
                    || self.identity.contains(&bone)
                {
                    Mat3::identity()
                } else {
                    special_procrustes(&rotation(&h))
                };
                write4(
                    &rigid(&r, &Vec3::zeros()),
                    &mut pose.data[index..index + 16],
                );
            }
        }
        let neutral = self.evaluate(&State {
            pose: pose.clone(),
            phenotypes: state.phenotypes.clone(),
        })?;
        let vn = neutral.get("vertices")?;
        let va = output_abs.get("vertices")?;
        let size = self.model.data.vertex_count();
        let weights = vec![1.; size];
        let mut vhat = self.vertices(&neutral)?;
        for bi in 0..b {
            let a = vn.data[bi * size * 3..(bi + 1) * size * 3]
                .chunks_exact(3)
                .map(vec3)
                .collect::<Vec<_>>();
            let z = va.data[bi * size * 3..(bi + 1) * size * 3]
                .chunks_exact(3)
                .map(vec3)
                .collect::<Vec<_>>();
            let t = rigid_registration(&a, &z, &weights, true)?;
            write4(&t, &mut pose.data[bi * j * 16..bi * j * 16 + 16]);
            for v in 0..n {
                let ix = (bi * n + v) * 3;
                let p = point(&t, &vec3(&vhat.data[ix..ix + 3]));
                vhat.data[ix..ix + 3].copy_from_slice(p.as_slice());
            }
        }
        Ok((pose, vhat))
    }
    fn optimize_shape(
        &self,
        state: &mut State,
        target: &Tensor,
        vhat: &Tensor,
        keys: &[usize],
        options: &FitOptions,
    ) -> Result<()> {
        if keys.is_empty() {
            return Ok(());
        }
        let b = target.shape[0];
        let n = self.unique.len();
        let d = state.phenotypes.shape[1];
        let count = keys.len();
        let rows = self.points.len() * 3;
        let mut jacobians = vec![DMatrix::<f64>::zeros(rows, count); b];
        for (col, &key) in keys.iter().enumerate() {
            let mut plus = state.clone();
            let mut minus = state.clone();
            let mut denominators = Vec::new();
            for bi in 0..b {
                let v = state.phenotypes.data[bi * d + key];
                let a = (v + self.options.eps).clamp(0.01, 0.99);
                let z = (v - self.options.eps).clamp(0.01, 0.99);
                plus.phenotypes.data[bi * d + key] = a;
                minus.phenotypes.data[bi * d + key] = z;
                denominators.push((a - z).max(1e-8));
            }
            let vp = self.vertices(&self.evaluate(&plus)?)?;
            let vm = self.vertices(&self.evaluate(&minus)?)?;
            for bi in 0..b {
                for (r, &v) in self.points.iter().enumerate() {
                    for axis in 0..3 {
                        let index = (bi * n + v) * 3 + axis;
                        jacobians[bi][(r * 3 + axis, col)] =
                            (vp.data[index] - vm.data[index]) / denominators[bi];
                    }
                }
            }
        }
        let mut deltas = vec![DVector::<f64>::zeros(count); b];
        for bi in 0..b {
            let mut residual = DVector::zeros(rows);
            for (r, &v) in self.points.iter().enumerate() {
                for axis in 0..3 {
                    let i = (bi * n + v) * 3 + axis;
                    residual[r * 3 + axis] = target.data[i] - vhat.data[i];
                }
            }
            let a = &jacobians[bi];
            let mut h = a.transpose() * a;
            for (i, &key) in keys.iter().enumerate() {
                let label = &self.model.phenotype_labels[key];
                let reg = self
                    .options
                    .regularization
                    .get(label)
                    .copied()
                    .unwrap_or_else(|| default_regularization(label));
                h[(i, i)] += reg;
            }
            let rhs = a.transpose() * residual;
            deltas[bi] = h.lu().solve(&rhs).ok_or_else(|| {
                Error::Invalid("singular phenotype normal equations; add regularization".into())
            })?;
        }
        if options.shared_phenotypes {
            let mut mean = DVector::zeros(count);
            for delta in &deltas {
                mean += delta / (b as f64);
            }
            for delta in &mut deltas {
                *delta = mean.clone();
            }
        }
        for (bi, values) in deltas.iter().enumerate() {
            for (i, &key) in keys.iter().enumerate() {
                let delta = if values[i].is_finite() {
                    values[i].clamp(-options.max_delta, options.max_delta)
                } else {
                    0.
                };
                state.phenotypes.data[bi * d + key] =
                    (state.phenotypes.data[bi * d + key] + delta).clamp(0.01, 0.99);
            }
        }
        Ok(())
    }
    fn fit_one(&self, target: &Tensor, options: &FitOptions) -> Result<State> {
        let b = target.shape[0];
        let mut state = self.initial(options, b)?;
        let out = self.evaluate(&state)?;
        let v = self.vertices(&out)?;
        let n = self.unique.len();
        let weights = vec![1.; n];
        let j = self.model.data.bone_count();
        for bi in 0..b {
            let a = v.data[bi * n * 3..(bi + 1) * n * 3]
                .chunks_exact(3)
                .map(vec3)
                .collect::<Vec<_>>();
            let z = target.data[bi * n * 3..(bi + 1) * n * 3]
                .chunks_exact(3)
                .map(vec3)
                .collect::<Vec<_>>();
            let transform = rigid_registration(&a, &z, &weights, true)?;
            write4(
                &transform,
                &mut state.pose.data[bi * j * 16..bi * j * 16 + 16],
            );
        }
        let keys = self
            .model
            .phenotype_labels
            .iter()
            .enumerate()
            .filter(|(_, s)| !options.excluded_phenotypes.contains(s))
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        let iterations = options.max_n_iters.unwrap_or(self.options.max_n_iters);
        let mut out = self.evaluate(&state)?;
        for iteration in 0..iterations {
            let (pose, vhat) = self.jointwise(&out, target, &state)?;
            state.pose = pose;
            if options.optimize_phenotypes {
                self.optimize_shape(&mut state, target, &vhat, &keys, options)?;
                if iteration + 1 == iterations {
                    state.pose = self.jointwise(&out, target, &state)?.0;
                }
            }
            out = self.evaluate(&state)?;
        }
        Ok(state)
    }
    /// Target may cover all model vertices, or just the sorted vertices referenced
    /// by faces (the Python inverter's target layout). Pose output uses model config.
    pub fn fit(&self, target: &Tensor, options: &FitOptions) -> Result<FitResult> {
        ensure(
            target.shape.len() == 3 && target.shape[0] > 0 && target.shape[2] == 3,
            "fit target must be [B,V,3]",
        )?;
        target.validate()?;
        ensure(
            options.max_delta > 0. && options.max_delta.is_finite(),
            "max_delta must be positive",
        )?;
        if options.post_gd {
            ensure(
                options.post_gd_steps <= 10_000
                    && options.post_gd_lr.is_finite()
                    && options.post_gd_lr > 0.,
                "invalid post_gd steps/learning rate",
            )?;
            ensure(
                options.post_gd_prior_weight.is_finite() && options.post_gd_prior_weight >= 0.,
                "invalid post_gd prior weight",
            )?;
            if options.post_gd_prior_weight > 0. {
                ensure(
                    self.options.shape_prior.is_some(),
                    "post_gd prior requires supplied shape calibration",
                )?;
                ensure(
                    options.optimize_phenotypes,
                    "post_gd prior requires phenotype optimization",
                )?;
            }
        }
        for key in &options.excluded_phenotypes {
            ensure(
                self.model.phenotype_labels.contains(key),
                format!("unknown excluded phenotype {key}"),
            )?;
        }
        let target = if target.shape[1] == self.model.data.vertex_count() {
            target.select(1, &self.unique)?
        } else {
            ensure(
                target.shape[1] == self.unique.len(),
                "target topology/vertex count mismatch",
            )?;
            target.clone()
        };
        let mut candidates = vec![options.clone()];
        for (key, values) in &options.multistart {
            ensure(
                self.model.phenotype_labels.contains(key)
                    && !values.is_empty()
                    && values.iter().all(|v| v.is_finite()),
                format!("invalid multistart values for {key}"),
            )?;
            ensure(
                candidates.len().saturating_mul(values.len()) <= 256,
                "multistart limited to 256 candidate configurations",
            )?;
            let mut expanded = Vec::new();
            for candidate in &candidates {
                for value in values {
                    let mut c = candidate.clone();
                    if c.initial_phenotype_kwargs.is_null() {
                        c.initial_phenotype_kwargs = json!({});
                    }
                    let map = c.initial_phenotype_kwargs.as_object_mut().ok_or_else(|| {
                        Error::Invalid("multistart needs named initial phenotypes".into())
                    })?;
                    map.insert(key.clone(), json!(value));
                    expanded.push(c);
                }
            }
            candidates = expanded;
        }
        let mut best: Option<(State, Vec<f64>)> = None;
        let b = target.shape[0];
        for candidate in candidates {
            let state = self.fit_one(&target, &candidate)?;
            let e = vertex_errors(&self.vertices(&self.evaluate(&state)?)?, &target)?;
            if let Some((previous, errors)) = &mut best {
                let shared =
                    options.shared_phenotypes && e.iter().sum::<f64>() < errors.iter().sum::<f64>();
                for bi in 0..b {
                    if shared || (!options.shared_phenotypes && e[bi] < errors[bi]) {
                        copy_batch(&state.pose, bi, &mut previous.pose, bi);
                        copy_batch(&state.phenotypes, bi, &mut previous.phenotypes, bi);
                        errors[bi] = e[bi];
                    }
                }
            } else {
                best = Some((state, e));
            }
        }
        let (state, mut mean_vertex_error) = best.unwrap();
        let mut output = self.evaluate(&state)?;
        let mut vertices = self.vertices(&output)?;
        let mut parameters = self.parameters(&state);
        parameters.pose_parameters = self
            .model
            .pose_parameters(&output, self.model.config.pose_parameterization)?
            .nested_json();
        parameters.pose_parameterization = Some(self.model.config.pose_parameterization);
        // Named phenotypes make fits usable with configuration changes and FFI callers.
        parameters.phenotype_kwargs = json!(self
            .model
            .phenotype_labels
            .iter()
            .enumerate()
            .map(|(i, k)| (
                k.clone(),
                (0..b)
                    .map(|bi| state.phenotypes.data[bi * state.phenotypes.shape[1] + i])
                    .collect::<Vec<_>>()
            ))
            .collect::<BTreeMap<_, _>>());
        let mut post_gd_losses = Vec::new();
        if options.post_gd {
            let mut selected = crate::differentiation::ParameterSelection {
                phenotypes: self
                    .model
                    .phenotype_labels
                    .iter()
                    .filter(|n| {
                        options.optimize_phenotypes && !options.excluded_phenotypes.contains(n)
                    })
                    .cloned()
                    .collect(),
                bone_rotations: self
                    .partition
                    .iter()
                    .enumerate()
                    .filter(|(i, entries)| {
                        *i == 0 || (!entries.is_empty() && !self.identity.contains(i))
                    })
                    .map(|(i, _)| self.model.data.metadata.bone_labels[i].clone())
                    .collect(),
                bone_translations: vec![self.model.data.metadata.bone_labels[0].clone()],
                ..Default::default()
            };
            if options.post_gd_optimize_local_changes {
                selected.local_changes = self.model.local_change_labels.clone();
            }
            if options.post_gd_optimize_facial_actions {
                selected.facial_actions = self.model.facial_action_labels.clone();
            }
            let refined = crate::refinement::refine(
                self.model,
                &target,
                &parameters,
                &crate::refinement::RefinementOptions {
                    steps: options.post_gd_steps,
                    learning_rate: options.post_gd_lr,
                    prior_weight: options.post_gd_prior_weight,
                    selection: Some(selected),
                    shared_phenotypes: options.shared_phenotypes,
                    ..Default::default()
                },
                self.options.shape_prior.as_ref(),
            )?;
            parameters = refined.parameters;
            output = refined.output;
            vertices = self.vertices(&output)?;
            mean_vertex_error = vertex_errors(&vertices, &target)?;
            post_gd_losses = refined.losses;
            let ph = model::parse_values(
                &parameters.phenotype_kwargs,
                &self.model.phenotype_labels,
                0.5,
                "refined phenotypes",
            )?;
            parameters.phenotype_kwargs = json!(self
                .model
                .phenotype_labels
                .iter()
                .enumerate()
                .map(|(i, n)| (
                    n.clone(),
                    (0..b)
                        .map(|bi| ph.data[bi * ph.shape[1] + i])
                        .collect::<Vec<_>>()
                ))
                .collect::<BTreeMap<_, _>>());
        }
        Ok(FitResult {
            parameters,
            vertices,
            output,
            mean_vertex_error,
            iterations: options.max_n_iters.unwrap_or(self.options.max_n_iters),
            post_gd_losses,
        })
    }
}
fn default_regularization(name: &str) -> f64 {
    match name {
        "age" => 10.,
        "height" => 1e-3,
        "cupsize" | "firmness" => 2.,
        "african" | "asian" | "caucasian" => 100.,
        _ => 1.,
    }
}
fn repeat_batch(t: &Tensor, b: usize) -> Result<Tensor> {
    model::broadcast(&[b, t.shape[0]])?;
    let mut shape = t.shape.clone();
    shape[0] = b;
    let mut out = Tensor::zeros(shape);
    let n = t.data.len() / t.shape[0];
    for bi in 0..b {
        out.data[bi * n..(bi + 1) * n]
            .copy_from_slice(&t.data[(bi % t.shape[0]) * n..(bi % t.shape[0] + 1) * n]);
    }
    Ok(out)
}
fn copy_batch(source: &Tensor, si: usize, dest: &mut Tensor, di: usize) {
    let n = source.data.len() / source.shape[0];
    dest.data[di * n..(di + 1) * n].copy_from_slice(&source.data[si * n..(si + 1) * n]);
}
pub fn vertex_errors(a: &Tensor, b: &Tensor) -> Result<Vec<f64>> {
    a.expect_shape(&b.shape, "vertex error tensors")?;
    ensure(
        a.shape.len() == 3 && a.shape[2] == 3 && a.shape[1] > 0,
        "vertex errors need [B,V,3]",
    )?;
    let n = a.shape[1];
    Ok((0..a.shape[0])
        .map(|bi| {
            (0..n)
                .map(|v| {
                    let i = (bi * n + v) * 3;
                    (vec3(&a.data[i..i + 3]) - vec3(&b.data[i..i + 3])).norm()
                })
                .sum::<f64>()
                / n as f64
        })
        .collect())
}
