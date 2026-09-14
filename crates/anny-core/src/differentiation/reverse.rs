//! Jacobian-transpose products assembled from the analytic directional rules.
//! This is parameter-level differentiation, not a general-purpose autograd tape.
use super::{jvp_at, validate_names, ParameterDirection};
use crate::{ensure, Anny, ModelOutput, Parameters, Result, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Explicit differentiated controls. An empty list means none, not all.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ParameterSelection {
    pub phenotypes: Vec<String>,
    pub local_changes: Vec<String>,
    pub facial_actions: Vec<String>,
    pub bone_rotations: Vec<String>,
    pub bone_translations: Vec<String>,
}
impl ParameterSelection {
    pub fn all(model: &Anny) -> Self {
        Self {
            phenotypes: model.phenotype_labels.clone(),
            local_changes: model.local_change_labels.clone(),
            facial_actions: model.facial_action_labels.clone(),
            bone_rotations: model.data.metadata.bone_labels.clone(),
            bone_translations: model.data.metadata.bone_labels.clone(),
        }
    }
    pub fn validate(&self, model: &Anny) -> Result<()> {
        for (names, allowed) in [
            (&self.phenotypes, &model.phenotype_labels),
            (&self.local_changes, &model.local_change_labels),
            (&self.facial_actions, &model.facial_action_labels),
            (&self.bone_rotations, &model.data.metadata.bone_labels),
            (&self.bone_translations, &model.data.metadata.bone_labels),
        ] {
            ensure(
                names.iter().collect::<BTreeSet<_>>().len() == names.len(),
                "duplicate differentiated control",
            )?;
            validate_names(&names.iter().map(|n| (n.clone(), ())).collect(), allowed)?;
        }
        Ok(())
    }
}

/// Compute J^T u for named output cotangents and selected input controls.
/// Rotation components are left-multiplicative infinitesimal radians, matching
/// `jvp`; translations are independent additive meters. Only one character is
/// accepted. No model perturbations or finite-difference epsilon are used.
///
/// The reference implementation contracts one analytic JVP per selected scalar,
/// so cost scales with the number of controls. This is not a reverse-mode tape
/// or a claim of optimized large-parameter fitting performance. At piecewise
/// boundaries the documented upstream subgradient convention of `jvp` applies.
pub fn vjp(
    model: &Anny,
    parameters: &Parameters,
    selection: &ParameterSelection,
    cotangents: &BTreeMap<String, Tensor>,
) -> Result<ParameterDirection> {
    let base = model.forward(parameters)?;
    vjp_at(model, parameters, selection, cotangents, &base)
}
pub(crate) fn vjp_at(
    model: &Anny,
    parameters: &Parameters,
    selection: &ParameterSelection,
    cotangents: &BTreeMap<String, Tensor>,
    base: &ModelOutput,
) -> Result<ParameterDirection> {
    selection.validate(model)?;
    ensure(
        base.get("vertices")?.shape[0] == 1,
        "vjp expects one character",
    )?;
    ensure(
        !cotangents.is_empty(),
        "vjp needs at least one output cotangent",
    )?;
    // Validate even with an empty selection, so malformed input never silently passes.
    let zero = jvp_at(model, parameters, &ParameterDirection::default(), base)?;
    for (name, cotangent) in cotangents {
        cotangent.expect_shape(&zero.get(name)?.shape, name)?;
    }
    let contract = |direction: &ParameterDirection| -> Result<f64> {
        let tangent = jvp_at(model, parameters, direction, base)?;
        let mut gradient = 0.;
        for (name, cotangent) in cotangents {
            gradient += tangent
                .get(name)?
                .data
                .iter()
                .zip(&cotangent.data)
                .map(|(a, b)| a * b)
                .sum::<f64>();
        }
        ensure(gradient.is_finite(), "non-finite contracted gradient")?;
        Ok(gradient)
    };
    let mut result = ParameterDirection::default();
    for (group, names) in [
        &selection.phenotypes,
        &selection.local_changes,
        &selection.facial_actions,
    ]
    .iter()
    .enumerate()
    {
        for name in *names {
            let mut direction = ParameterDirection::default();
            match group {
                0 => &mut direction.phenotypes,
                1 => &mut direction.local_changes,
                _ => &mut direction.facial_actions,
            }
            .insert(name.clone(), 1.);
            let gradient = contract(&direction)?;
            match group {
                0 => &mut result.phenotypes,
                1 => &mut result.local_changes,
                _ => &mut result.facial_actions,
            }
            .insert(name.clone(), gradient);
        }
    }
    for (rotations, names) in [
        (true, &selection.bone_rotations),
        (false, &selection.bone_translations),
    ] {
        for name in names {
            let mut gradient = [0.; 3];
            for axis in 0..3 {
                let mut direction = ParameterDirection::default();
                let mut basis = [0.; 3];
                basis[axis] = 1.;
                if rotations {
                    &mut direction.bone_rotations
                } else {
                    &mut direction.bone_translations
                }
                .insert(name.clone(), basis);
                gradient[axis] = contract(&direction)?;
            }
            if rotations {
                &mut result.bone_rotations
            } else {
                &mut result.bone_translations
            }
            .insert(name.clone(), gradient);
        }
    }
    Ok(result)
}
