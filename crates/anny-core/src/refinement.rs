//! Native optional post-fit Adam refinement. Uses analytic parameter derivatives,
//! never finite differences or an interpreter. The reference solver is f64.
use crate::{
    differentiation::{vjp_at, ParameterDirection, ParameterSelection},
    distribution::SimpleShapeDistribution,
    ensure,
    math::*,
    model,
    prior::shape_prior_gradient,
    Anny, ModelOutput, Parameters, PoseParameterization, Result, Tensor,
};
use nalgebra::UnitQuaternion;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum ReconstructionLoss {
    #[default]
    MeanSquare,
    /// Coordinate-wise Huber loss, normalized over batch, vertices and xyz.
    Huber { delta: f64 },
}
impl ReconstructionLoss {
    fn value_derivative(&self, residual: f64) -> (f64, f64) {
        match *self {
            Self::MeanSquare => (residual * residual, 2. * residual),
            Self::Huber { delta } if residual.abs() > delta => (
                delta * (residual.abs() - delta * 0.5),
                delta * residual.signum(),
            ),
            Self::Huber { .. } => (0.5 * residual * residual, residual),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RefinementOptions {
    pub steps: usize,
    pub learning_rate: f64,
    pub prior_weight: f64,
    /// None selects phenotypes, weighted bones plus the root, and root translation.
    /// Some(empty selection) fixes all controls. Local/facial controls are opt-in.
    pub selection: Option<ParameterSelection>,
    pub shared_phenotypes: bool,
    pub loss: ReconstructionLoss,
}
impl Default for RefinementOptions {
    fn default() -> Self {
        Self {
            steps: 100,
            learning_rate: 1e-3,
            prior_weight: 0.,
            selection: None,
            shared_phenotypes: false,
            loss: ReconstructionLoss::MeanSquare,
        }
    }
}
pub struct RefinementResult {
    pub parameters: Parameters,
    pub output: ModelOutput,
    /// Initial objective followed by the objective after every Adam step.
    pub losses: Vec<f64>,
    pub iterations: usize,
}
impl RefinementResult {
    pub fn to_json(&self) -> serde_json::Value {
        json!({"parameters":self.parameters,"losses":self.losses,"iterations":self.iterations})
    }
}
pub fn default_selection(model: &Anny) -> Result<ParameterSelection> {
    let weights = model.data.get("vertex_bone_weights")?;
    let indices = model.data.get("vertex_bone_indices")?;
    let mut active = BTreeSet::from([0usize]);
    for (&weight, &bone) in weights.data.iter().zip(&indices.data) {
        if weight > 0. {
            active.insert(bone as usize);
        }
    }
    Ok(ParameterSelection {
        phenotypes: model.phenotype_labels.clone(),
        bone_rotations: active
            .into_iter()
            .map(|i| model.data.metadata.bone_labels[i].clone())
            .collect(),
        bone_translations: vec![model.data.metadata.bone_labels[0].clone()],
        ..Default::default()
    })
}
fn sigmoid(x: f64) -> f64 {
    if x >= 0. {
        1. / (1. + (-x).exp())
    } else {
        let e = x.exp();
        e / (1. + e)
    }
}
fn logit(x: f64) -> f64 {
    let x = x.clamp(1e-6, 1. - 1e-6);
    (x / (1. - x)).ln()
}
fn left_jacobian(w: Vec3) -> Mat3 {
    let t2 = w.norm_squared();
    let h = Mat3::new(0., -w.z, w.y, w.z, 0., -w.x, -w.y, w.x, 0.);
    let (a, b) = if t2 < 1e-6 {
        (
            0.5 - t2 / 24. + t2 * t2 / 720.,
            1. / 6. - t2 / 120. + t2 * t2 / 5040.,
        )
    } else {
        let t = t2.sqrt();
        ((1. - t.cos()) / t2, (t - t.sin()) / (t2 * t))
    };
    Mat3::identity() + h * a + h * h * b
}
#[derive(Clone)]
struct State {
    ph: Tensor,
    lo: Tensor,
    fa: Tensor,
    pose: Tensor,
    rotvecs: Vec<Vec3>,
}
fn repeated(t: Tensor, batch: usize) -> Result<Tensor> {
    model::broadcast(&[batch, t.shape[0]])?;
    let mut shape = t.shape.clone();
    shape[0] = batch;
    let mut out = Tensor::zeros(shape);
    let width = t.data.len() / t.shape[0];
    for b in 0..batch {
        out.data[b * width..(b + 1) * width]
            .copy_from_slice(&t.data[(b % t.shape[0]) * width..(b % t.shape[0] + 1) * width]);
    }
    Ok(out)
}
impl State {
    fn new(model: &Anny, p: &Parameters, batch: usize) -> Result<Self> {
        let out = model.forward(p)?;
        let pose = repeated(
            model.pose_parameters(&out, PoseParameterization::LocalBone)?,
            batch,
        )?;
        let mut state = Self {
            ph: repeated(
                model::parse_values(
                    &p.phenotype_kwargs,
                    &model.phenotype_labels,
                    0.5,
                    "phenotypes",
                )?,
                batch,
            )?,
            lo: repeated(
                model::parse_values(
                    &p.local_changes_kwargs,
                    &model.local_change_labels,
                    0.,
                    "local changes",
                )?,
                batch,
            )?,
            fa: repeated(
                model::parse_values(
                    &p.facial_actions,
                    &model.facial_action_labels,
                    0.,
                    "facial actions",
                )?,
                batch,
            )?,
            rotvecs: Vec::new(),
            pose,
        };
        let bones = model.data.bone_count();
        for (i, row) in state.pose.data.chunks_exact_mut(16).enumerate() {
            let mut h = mat4(row);
            let r = rotation(&h);
            ensure(
                (r.transpose() * r - Mat3::identity()).norm() < 1e-4
                    && (r.determinant() - 1.).abs() < 1e-4,
                "refinement requires proper rigid input rotations",
            )?;
            state
                .rotvecs
                .push(UnitQuaternion::from_matrix(&r).scaled_axis());
            // Match upstream post_gd: joint translations are zero, root is free.
            if i % bones != 0 {
                for a in 0..3 {
                    h[(a, 3)] = 0.;
                }
            }
            write4(&h, row);
        }
        Ok(state)
    }
    fn parameters(&self, sample: Option<usize>) -> Parameters {
        let slice = |t: &Tensor| {
            if let Some(b) = sample {
                let n = t.data.len() / t.shape[0];
                let mut shape = t.shape.clone();
                shape[0] = 1;
                Tensor {
                    shape,
                    data: t.data[b * n..(b + 1) * n].to_vec(),
                    kind: t.kind,
                }
                .nested_json()
            } else {
                t.nested_json()
            }
        };
        Parameters {
            phenotype_kwargs: slice(&self.ph),
            local_changes_kwargs: slice(&self.lo),
            facial_actions: slice(&self.fa),
            pose_parameters: slice(&self.pose),
            pose_parameterization: Some(PoseParameterization::LocalBone),
            ..Default::default()
        }
    }
}
#[derive(Clone, Copy)]
enum Control {
    Phenotype(usize),
    Local(usize),
    Face(usize),
    Rotation(usize, usize),
    Translation(usize),
}
struct Variable {
    control: Control,
    sample: Option<usize>,
}
fn variables(
    model: &Anny,
    selection: &ParameterSelection,
    batch: usize,
    shared: bool,
) -> Vec<Variable> {
    let mut result = Vec::new();
    let mut add = |control, share| {
        for b in 0..if share { 1 } else { batch } {
            result.push(Variable {
                control,
                sample: if share { None } else { Some(b) },
            });
        }
    };
    for (names, labels, kind) in [
        (&selection.phenotypes, &model.phenotype_labels, 0),
        (&selection.local_changes, &model.local_change_labels, 1),
        (&selection.facial_actions, &model.facial_action_labels, 2),
    ] {
        for name in names {
            let i = labels.iter().position(|x| x == name).unwrap();
            add(
                match kind {
                    0 => Control::Phenotype(i),
                    1 => Control::Local(i),
                    _ => Control::Face(i),
                },
                kind == 0 && shared,
            );
        }
    }
    for name in &selection.bone_rotations {
        let i = model
            .data
            .metadata
            .bone_labels
            .iter()
            .position(|x| x == name)
            .unwrap();
        for axis in 0..3 {
            add(Control::Rotation(i, axis), false);
        }
    }
    if !selection.bone_translations.is_empty() {
        for axis in 0..3 {
            add(Control::Translation(axis), false);
        }
    }
    result
}
impl Variable {
    fn value(&self, s: &State) -> f64 {
        let b = self.sample.unwrap_or(0);
        match self.control {
            Control::Phenotype(i) => logit(s.ph.data[b * s.ph.shape[1] + i]),
            Control::Local(i) => s.lo.data[b * s.lo.shape[1] + i],
            Control::Face(i) => s.fa.data[b * s.fa.shape[1] + i],
            Control::Rotation(i, a) => s.rotvecs[b * s.pose.shape[1] + i][a],
            Control::Translation(a) => s.pose.data[b * s.pose.shape[1] * 16 + a * 4 + 3],
        }
    }
    fn set(&self, s: &mut State, x: f64) {
        let batch = s.ph.shape[0];
        for b in 0..batch {
            if self.sample.is_some_and(|i| i != b) {
                continue;
            }
            match self.control {
                Control::Phenotype(i) => s.ph.data[b * s.ph.shape[1] + i] = sigmoid(x),
                Control::Local(i) => s.lo.data[b * s.lo.shape[1] + i] = x,
                Control::Face(i) => s.fa.data[b * s.fa.shape[1] + i] = x,
                Control::Rotation(i, a) => s.rotvecs[b * s.pose.shape[1] + i][a] = x,
                Control::Translation(a) => s.pose.data[b * s.pose.shape[1] * 16 + a * 4 + 3] = x,
            }
        }
    }
    fn gradient(&self, model: &Anny, s: &State, g: &[ParameterDirection]) -> f64 {
        let mut out = 0.;
        for (b, gradient) in g.iter().enumerate() {
            if self.sample.is_some_and(|i| i != b) {
                continue;
            }
            out += match self.control {
                Control::Phenotype(i) => {
                    let p = s.ph.data[b * s.ph.shape[1] + i];
                    gradient.phenotypes[&model.phenotype_labels[i]] * p * (1. - p)
                }
                Control::Local(i) => gradient.local_changes[&model.local_change_labels[i]],
                Control::Face(i) => gradient.facial_actions[&model.facial_action_labels[i]],
                Control::Rotation(i, a) => {
                    (left_jacobian(s.rotvecs[b * s.pose.shape[1] + i]).transpose()
                        * vec3(&gradient.bone_rotations[&model.data.metadata.bone_labels[i]]))[a]
                }
                Control::Translation(a) => {
                    gradient.bone_translations[&model.data.metadata.bone_labels[0]][a]
                }
            };
        }
        out
    }
}
struct Objective<'a> {
    model: &'a Anny,
    target: &'a Tensor,
    ids: &'a [usize],
    selection: &'a ParameterSelection,
    options: &'a RefinementOptions,
    prior: Option<&'a SimpleShapeDistribution>,
}
impl Objective<'_> {
    fn evaluate(
        &self,
        state: &State,
        need_gradient: bool,
    ) -> Result<(f64, Vec<ParameterDirection>)> {
        let Self {
            model,
            target,
            ids,
            selection,
            options,
            prior,
        } = *self;
        let batch = target.shape[0];
        let normalizer = (batch * ids.len() * 3) as f64;
        let mut loss = 0.;
        let mut gradients = Vec::new();
        for b in 0..batch {
            let p = state.parameters(Some(b));
            let base = model.forward(&p)?;
            let values = base.get("vertices")?;
            let mut cot = Tensor::zeros(values.shape.clone());
            for (row, &id) in ids.iter().enumerate() {
                for axis in 0..3 {
                    let index = id * 3 + axis;
                    let residual =
                        values.data[index] - target.data[(b * ids.len() + row) * 3 + axis];
                    let (l, d) = options.loss.value_derivative(residual);
                    loss += l / normalizer;
                    cot.data[index] = d / normalizer;
                }
            }
            let mut grad = if need_gradient {
                vjp_at(
                    model,
                    &p,
                    selection,
                    &BTreeMap::from([("vertices".into(), cot)]),
                    &base,
                )?
            } else {
                ParameterDirection::default()
            };
            if options.prior_weight > 0. {
                let phenos = model
                    .phenotype_labels
                    .iter()
                    .enumerate()
                    .map(|(i, n)| (n.clone(), state.ph.data[b * state.ph.shape[1] + i]))
                    .collect();
                let (value, g) = shape_prior_gradient(prior.unwrap(), &phenos)?;
                let weight = options.prior_weight / batch as f64;
                loss += weight * value;
                if need_gradient {
                    for (name, value) in &mut grad.phenotypes {
                        *value += weight * g.get(name).copied().unwrap_or(0.);
                    }
                }
            }
            gradients.push(grad);
        }
        ensure(loss.is_finite(), "non-finite refinement objective")?;
        Ok((loss, gradients))
    }
}
/// Refine a known-correspondence target with native Adam and analytic gradients.
/// Target order is all model vertices OR the sorted face-referenced vertex set.
/// Input pose is converted to local-bone, and output pose to the model convention.
/// Shared phenotypes are optimized as one common vector across the batch.
///
/// The default objective and parameterization match upstream post_gd: mean square
/// vertex residual, rotation vectors, root translation, sigmoid phenotype logits,
/// optional calibrated prior and local/facial clamps. Optimizer trajectories need
/// not be bit-identical to PyTorch. The supplied calibration stays explicit.
pub fn refine(
    model: &Anny,
    target: &Tensor,
    initial: &Parameters,
    options: &RefinementOptions,
    prior: Option<&SimpleShapeDistribution>,
) -> Result<RefinementResult> {
    target.validate()?;
    ensure(
        target.shape.len() == 3
            && target.shape[0] > 0
            && target.shape[0] <= 10_000
            && target.shape[2] == 3,
        "refinement target must be [B,V,3]",
    )?;
    ensure(
        options.steps <= 10_000 && options.learning_rate.is_finite() && options.learning_rate > 0.,
        "invalid refinement steps/learning rate",
    )?;
    ensure(
        options.prior_weight.is_finite() && options.prior_weight >= 0.,
        "invalid shape-prior weight",
    )?;
    if let ReconstructionLoss::Huber { delta } = options.loss {
        ensure(
            delta.is_finite() && delta > 0.,
            "Huber delta must be finite and positive",
        )?;
    }
    let selection = options
        .selection
        .clone()
        .map_or_else(|| default_selection(model), Ok)?;
    selection.validate(model)?;
    ensure(
        selection
            .bone_translations
            .iter()
            .all(|n| n == &model.data.metadata.bone_labels[0]),
        "post_gd only optimizes root translation",
    )?;
    if options.prior_weight > 0. {
        let prior = prior.ok_or_else(|| {
            crate::Error::Invalid("positive prior weight needs supplied shape calibration".into())
        })?;
        ensure(
            !selection.phenotypes.is_empty(),
            "shape prior requires optimized phenotypes",
        )?;
        ensure(
            prior.phenotype_labels == model.phenotype_labels,
            "shape calibration labels do not match model",
        )?;
    }
    let ids: Vec<_> = model
        .data
        .get("faces")?
        .checked_indices(model.data.vertex_count(), "refinement faces")?
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    ensure(
        !ids.is_empty(),
        "refinement needs referenced surface vertices",
    )?;
    let target = if target.shape[1] == model.data.vertex_count() {
        target.select(1, &ids)?
    } else {
        ensure(
            target.shape[1] == ids.len(),
            "refinement target topology mismatch",
        )?;
        target.clone()
    };
    let batch = target.shape[0];
    let mut state = State::new(model, initial, batch)?;
    if options.shared_phenotypes {
        let width = state.ph.shape[1];
        for b in 1..batch {
            ensure(
                state.ph.data[..width]
                    .iter()
                    .zip(&state.ph.data[b * width..(b + 1) * width])
                    .all(|(a, b)| (a - b).abs() < 1e-12),
                "shared refinement requires equal initial phenotypes",
            )?;
        }
    }
    let variables = variables(model, &selection, batch, options.shared_phenotypes);
    ensure(variables.len() <= 100_000, "too many refinement variables")?;
    let mut x: Vec<_> = variables.iter().map(|v| v.value(&state)).collect();
    let mut first = vec![0.; x.len()];
    let mut second = vec![0.; x.len()];
    let mut losses = Vec::new();
    for step in 0..=options.steps {
        for (v, &x) in variables.iter().zip(&x) {
            v.set(&mut state, x);
        }
        for b in 0..batch {
            for name in &selection.bone_rotations {
                let i = model
                    .data
                    .metadata
                    .bone_labels
                    .iter()
                    .position(|n| n == name)
                    .unwrap();
                let index = b * model.data.bone_count() + i;
                let rotation =
                    UnitQuaternion::from_scaled_axis(state.rotvecs[index]).to_rotation_matrix();
                let mut h = mat4(&state.pose.data[index * 16..(index + 1) * 16]);
                h.fixed_view_mut::<3, 3>(0, 0).copy_from(rotation.matrix());
                write4(&h, &mut state.pose.data[index * 16..(index + 1) * 16]);
            }
        }
        let (loss, gradient) = Objective {
            model,
            target: &target,
            ids: &ids,
            selection: &selection,
            options,
            prior,
        }
        .evaluate(&state, step < options.steps)?;
        losses.push(loss);
        if step == options.steps {
            break;
        }
        for (i, v) in variables.iter().enumerate() {
            let g = v.gradient(model, &state, &gradient);
            ensure(g.is_finite(), "non-finite optimizer gradient")?;
            first[i] = 0.9 * first[i] + 0.1 * g;
            second[i] = 0.999 * second[i] + 0.001 * g * g;
            let m = first[i] / (1. - 0.9f64.powi((step + 1) as i32));
            let vhat = second[i] / (1. - 0.999f64.powi((step + 1) as i32));
            x[i] -= options.learning_rate * m / (vhat.sqrt() + 1e-8);
            match v.control {
                Control::Local(_) => x[i] = x[i].clamp(-1., 1.),
                Control::Face(_) => x[i] = x[i].clamp(0., 1.),
                _ => {}
            }
            ensure(x[i].is_finite(), "non-finite Adam update")?;
        }
    }
    let mut parameters = state.parameters(None);
    let output = model.forward(&parameters)?;
    parameters.pose_parameters = model
        .pose_parameters(&output, model.config.pose_parameterization)?
        .nested_json();
    parameters.pose_parameterization = Some(model.config.pose_parameterization);
    Ok(RefinementResult {
        parameters,
        output,
        losses,
        iterations: options.steps,
    })
}
