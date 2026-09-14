// Port of NAVER Anny ModelData, phenotype and rigged model semantics.
// Copyright (C) 2025 NAVER Corp. SPDX-License-Identifier: Apache-2.0
use crate::{config::*, ensure, math::*, tensor::Archive, Error, Result, Tensor, DATA_VERSION};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub bone_labels: Vec<String>,
    pub bone_parents: Vec<i32>,
    #[serde(default)]
    pub blendshape_labels: Vec<String>,
}
impl ModelMetadata {
    pub fn local_change_labels(&self) -> Vec<String> {
        let locals: Vec<_> = self
            .blendshape_labels
            .iter()
            .filter_map(|s| s.strip_prefix("local_change:"))
            .collect();
        locals.iter().step_by(2).map(|s| (*s).into()).collect()
    }
    pub fn facial_action_labels(&self) -> Vec<String> {
        self.blendshape_labels
            .iter()
            .filter_map(|s| s.strip_prefix("facial_action:").map(String::from))
            .collect()
    }
}
#[derive(Clone, Debug, Default)]
pub struct ModelData {
    pub metadata: ModelMetadata,
    pub arrays: BTreeMap<String, Tensor>,
}
/// The dtype-independent half of model validation: mask shape and binarity, the block lengths against
/// metadata, and the macro/facial/local ordering that the coefficient packing depends on.
///
/// `Anny::from_model_data` and `AnnyF32::from_bytes` both call this so a prepared payload is checked
/// by exactly the rules a freshly built model is, rather than the typed loader trusting its bytes.
pub(crate) fn validate_model_blocks(
    metadata: &ModelMetadata,
    blendshape_count: usize,
    mask_shape: &[usize],
    mask_is_binary: bool,
) -> Result<(Vec<String>, Vec<String>)> {
    ensure(
        mask_shape.len() == 2 && mask_shape[1] == 26,
        "phenotype mask must have 26 columns",
    )?;
    ensure(mask_is_binary, "phenotype mask must contain zero or one")?;
    let local_change_labels = metadata.local_change_labels();
    let facial_action_labels = metadata.facial_action_labels();
    ensure(
        mask_shape[0] + local_change_labels.len() * 2 + facial_action_labels.len()
            == blendshape_count,
        "blendshape blocks do not match metadata",
    )?;
    // Coefficient packing is part of the compatibility contract, not map order.
    let m = mask_shape[0];
    let f = facial_action_labels.len();
    ensure(
        metadata.blendshape_labels[..m]
            .iter()
            .all(|s| !s.starts_with("local_change:") && !s.starts_with("facial_action:"))
            && metadata.blendshape_labels[m..m + f]
                .iter()
                .all(|s| s.starts_with("facial_action:"))
            && metadata.blendshape_labels[m + f..]
                .iter()
                .all(|s| s.starts_with("local_change:")),
        "blendshape blocks must be macro, facial, then paired local",
    )?;
    Ok((local_change_labels, facial_action_labels))
}

impl ModelData {
    pub fn get(&self, name: &str) -> Result<&Tensor> {
        self.arrays
            .get(name)
            .ok_or_else(|| Error::Invalid(format!("ModelData is missing {name}")))
    }
    pub fn put(&mut self, name: &str, t: Tensor) {
        self.arrays.insert(name.into(), t);
    }
    pub fn remove(&mut self, names: &[&str]) {
        for n in names {
            self.arrays.remove(*n);
        }
    }
    pub fn from_archive(a: Archive) -> Result<Self> {
        let version = a
            .metadata
            .get("data_version")
            .and_then(|s| s.parse::<usize>().ok());
        ensure(
            version == Some(DATA_VERSION),
            format!("ModelData data_version {version:?}; expected {DATA_VERSION}"),
        )?;
        let m = a
            .metadata
            .get("metadata")
            .ok_or_else(|| Error::Invalid("ModelData missing metadata header".into()))?;
        let data = Self {
            metadata: serde_json::from_str(m)?,
            arrays: a.tensors,
        };
        data.validate()?;
        Ok(data)
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_archive(Archive::from_bytes(bytes)?)
    }
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_bytes(&std::fs::read(path)?)
    }
    pub fn archive(&self, config: Option<&AnnyConfig>) -> Result<Archive> {
        self.validate()?;
        let mut metadata = std::collections::HashMap::from([
            ("metadata".into(), serde_json::to_string(&self.metadata)?),
            ("metadata_class".into(), "ModelMetadata".into()),
            ("data_version".into(), DATA_VERSION.to_string()),
            ("anny_version".into(), "0.6.0".into()),
            ("anny_rust_upstream".into(), crate::UPSTREAM_REVISION.into()),
        ]);
        if let Some(c) = config {
            metadata.insert("anny_rust_config".into(), serde_json::to_string(c)?);
        }
        Ok(Archive {
            tensors: self.arrays.clone(),
            metadata,
        })
    }
    pub fn save(
        &self,
        path: impl AsRef<std::path::Path>,
        config: Option<&AnnyConfig>,
    ) -> Result<()> {
        self.archive(config)?.save(path)
    }
    pub fn vertex_count(&self) -> usize {
        self.arrays
            .get("template_vertices")
            .and_then(|t| t.shape.first())
            .copied()
            .unwrap_or(0)
    }
    pub fn bone_count(&self) -> usize {
        self.metadata.bone_labels.len()
    }
    pub fn blendshape_count(&self) -> usize {
        self.metadata.blendshape_labels.len()
    }
    pub fn validate(&self) -> Result<()> {
        let (n, j, c) = (
            self.vertex_count(),
            self.bone_count(),
            self.blendshape_count(),
        );
        ensure(n > 0 && j > 0, "model must have vertices and bones")?;
        ensure(
            self.metadata.bone_parents.len() == j,
            "bone metadata length mismatch",
        )?;
        let order = propagation_order(&self.metadata.bone_parents)?;
        ensure(
            order[0] == 0
                && self
                    .metadata
                    .bone_parents
                    .iter()
                    .filter(|&&p| p < 0)
                    .count()
                    == 1,
            "Anny model needs one root at index zero",
        )?;
        for (names, label) in [
            (&self.metadata.bone_labels, "bone"),
            (&self.metadata.blendshape_labels, "blendshape"),
        ] {
            ensure(
                names.iter().collect::<BTreeSet<_>>().len() == names.len(),
                format!("duplicate {label} label"),
            )?;
        }
        for (key, shape) in [
            ("template_vertices", vec![n, 3]),
            ("blendshapes", vec![c, n, 3]),
            ("template_bone_heads", vec![j, 3]),
            ("bone_heads_blendshapes", vec![c, j, 3]),
            ("base_mesh_vertex_indices", vec![n]),
        ] {
            self.get(key)?.expect_shape(&shape, key)?;
        }
        let faces = self.get("faces")?;
        ensure(
            faces.shape.len() == 2 && [3, 4].contains(&faces.shape[1]),
            "faces must be F x 3 or F x 4",
        )?;
        faces.checked_indices(n, "faces")?;
        let weights = self.get("vertex_bone_weights")?;
        ensure(
            weights.shape.len() == 2 && weights.shape[0] == n && weights.shape[1] > 0,
            "invalid skinning shape",
        )?;
        weights.validate()?;
        let ids = self.get("vertex_bone_indices")?;
        ids.expect_shape(&weights.shape, "vertex_bone_indices")?;
        ids.checked_indices(j, "bone indices")?;
        for row in weights.data.chunks_exact(weights.shape[1]) {
            ensure(
                (row.iter().sum::<f64>() - 1.).abs() < 1e-5,
                "skinning weights must sum to one",
            )?;
        }
        if let Some(uv) = self.arrays.get("texture_coordinates") {
            ensure(uv.shape.len() == 2 && uv.shape[1] == 2, "invalid UV shape")?;
            let ft = self.get("face_texture_coordinate_indices")?;
            ft.expect_shape(&faces.shape, "face UV indices")?;
            ft.checked_indices(uv.shape[0], "face UV indices")?;
        } else {
            ensure(
                !self.arrays.contains_key("face_texture_coordinate_indices"),
                "face UV indices without coordinates",
            )?;
        }
        for (name, t) in &self.arrays {
            t.validate()
                .map_err(|e| Error::Invalid(format!("{name}: {e}")))?;
        }
        for (name, shape) in [
            ("template_bone_tails", vec![j, 3]),
            ("bone_tails_blendshapes", vec![c, j, 3]),
            ("bone_template_orientation_matrices", vec![j, 3, 3]),
            ("bone_orientation_blendshapes", vec![c, j, 3, 3]),
            ("reference_bone_orientations", vec![j, 3, 3]),
        ] {
            if let Some(t) = self.arrays.get(name) {
                t.expect_shape(&shape, name)?;
            }
        }
        if let Some(t) = self.arrays.get("bone_rolls_rotmat") {
            ensure(
                t.shape == [j, 3, 3] || t.shape == [1, j, 3, 3],
                "invalid bone rolls shape",
            )?;
        }
        Ok(())
    }
}

/// Scalar/named-vector/stacked-array inputs use the same labels as Python Anny.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Parameters {
    pub phenotype_kwargs: Value,
    pub local_changes_kwargs: Value,
    pub facial_actions: Value,
    pub pose_parameters: Value,
    pub pose_parameterization: Option<PoseParameterization>,
    pub return_bone_ends: bool,
}
#[derive(Clone, Debug, Default)]
pub struct ModelOutput {
    pub arrays: BTreeMap<String, Tensor>,
}
impl ModelOutput {
    pub fn get(&self, name: &str) -> Result<&Tensor> {
        self.arrays
            .get(name)
            .ok_or_else(|| Error::Invalid(format!("missing output {name}")))
    }
    pub fn to_json(&self) -> Value {
        Value::Object(
            self.arrays
                .iter()
                .map(|(k, v)| (k.clone(), v.nested_json()))
                .collect(),
        )
    }
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        Archive {
            tensors: self.arrays.clone(),
            metadata: Default::default(),
        }
        .save(path)
    }
}
#[derive(Clone, Debug)]
pub struct Anny {
    pub data: ModelData,
    pub config: AnnyConfig,
    pub rig: RigConfig,
    pub phenotype_labels: Vec<String>,
    pub local_change_labels: Vec<String>,
    pub facial_action_labels: Vec<String>,
}
impl Anny {
    pub(crate) fn resolved_rig(&self) -> &RigConfig {
        &self.rig
    }
    pub fn from_model_data(data: ModelData, config: AnnyConfig) -> Result<Self> {
        data.validate()?;
        config.validate()?;
        let rig = config.rig.resolve()?;
        let mask = data.get("stacked_phenotype_blend_shapes_mask")?;
        let (local_change_labels, facial_action_labels) = validate_model_blocks(
            &data.metadata,
            data.blendshape_count(),
            &mask.shape,
            mask.data.iter().all(|&x| x == 0. || x == 1.),
        )?;
        validate_orientation(&data, rig.bone_orientation)?;
        let phenotype_labels = config.phenotype_labels();
        Ok(Self {
            data,
            config,
            rig,
            phenotype_labels,
            local_change_labels,
            facial_action_labels,
        })
    }
    pub fn from_bytes(bytes: &[u8], config: Option<AnnyConfig>) -> Result<Self> {
        let a = Archive::from_bytes(bytes)?;
        let c = match config {
            Some(c) => c,
            None => a
                .metadata
                .get("anny_rust_config")
                .map(|v| serde_json::from_str(v))
                .transpose()?
                .unwrap_or_default(),
        };
        Self::from_model_data(ModelData::from_archive(a)?, c)
    }
    pub fn load(path: impl AsRef<std::path::Path>, config: Option<AnnyConfig>) -> Result<Self> {
        Self::from_bytes(&std::fs::read(path)?, config)
    }
    pub fn describe(&self) -> Value {
        json!({"vertices":self.data.vertex_count(),"faces":self.data.arrays["faces"].shape[0],"bones":self.data.bone_count(),"blendshapes":self.data.blendshape_count(),"bone_labels":self.data.metadata.bone_labels,"bone_parents":self.data.metadata.bone_parents,"phenotype_labels":self.phenotype_labels,"local_change_labels":self.local_change_labels,"facial_action_labels":self.facial_action_labels,"config":self.config})
    }
    pub fn rest_model(&self, coeffs: &Tensor) -> Result<ModelOutput> {
        rest_model(&self.data, &self.rig, coeffs)
    }
    pub fn forward(&self, p: &Parameters) -> Result<ModelOutput> {
        let coeffs = self.coefficients(p)?;
        forward_model(
            &self.data,
            &self.rig,
            &coeffs,
            &p.pose_parameters,
            p.pose_parameterization
                .unwrap_or(self.config.pose_parameterization),
            self.config.skinning_method,
            p.return_bone_ends,
        )
    }
    pub fn pose_parameters(
        &self,
        output: &ModelOutput,
        mode: PoseParameterization,
    ) -> Result<Tensor> {
        pose_parameters(&self.data, output, mode)
    }
    /// Build a reusable pose-only evaluation context for a fixed parameter set.
    ///
    /// The phenotype/local-change/facial coefficients and the rest model both depend on the
    /// parameters but not on the pose, and [`Anny::forward`] rebuilds both on every call. A session
    /// evaluates them once, so [`PoseSession::update`] pays only for the pose-dependent half. Use
    /// this for animation, re-posing and editor sliders, where the pose changes and the rest of the
    /// parameters do not.
    ///
    /// The session fixes the pose parameterization, skinning method and bone-end setting taken from
    /// these parameters and the model config. Changing any of those, or the phenotype/local-change/
    /// facial selections, requires a new session; [`Anny::forward`] remains the general path.
    pub fn pose_session(&self, p: &Parameters) -> Result<PoseSession<'_>> {
        let coefficients = self.coefficients(p)?;
        let output = rest_model(&self.data, &self.rig, &coefficients)?;
        Ok(PoseSession {
            model: self,
            coefficients,
            output,
            pose_parameterization: p
                .pose_parameterization
                .unwrap_or(self.config.pose_parameterization),
            skinning: self.config.skinning_method,
            return_bone_ends: p.return_bone_ends,
        })
    }
}

/// A reusable pose-only evaluation context; see [`Anny::pose_session`].
///
/// The result of [`PoseSession::update`] is the same [`ModelOutput`] shape [`Anny::forward`] returns,
/// including the shared rest arrays, and is numerically identical to the corresponding
/// `forward` call for the same parameters.
pub struct PoseSession<'a> {
    model: &'a Anny,
    coefficients: Tensor,
    output: ModelOutput,
    pose_parameterization: PoseParameterization,
    skinning: SkinningMethod,
    return_bone_ends: bool,
}
impl PoseSession<'_> {
    /// The phenotype/local-change/facial coefficients this session was built with.
    pub fn coefficients(&self) -> &Tensor {
        &self.coefficients
    }
    /// The output of the most recent [`PoseSession::update`], or the rest model if no pose has been
    /// evaluated yet.
    pub fn output(&self) -> &ModelOutput {
        &self.output
    }
    /// Evaluate a pose against the cached rest model, reusing the previous output buffers.
    pub fn update(&mut self, pose: &Value) -> Result<&ModelOutput> {
        let output = std::mem::take(&mut self.output);
        match pose_model(
            &self.model.data,
            &self.model.rig,
            output,
            pose,
            self.pose_parameterization,
            self.skinning,
            self.return_bone_ends,
        ) {
            Ok(output) => {
                self.output = output;
                Ok(&self.output)
            }
            Err(e) => {
                // `pose_model` consumed the output it was handed, and the rest arrays live in it, so
                // a bad pose has to leave a rebuilt rest model behind to keep the session usable.
                // This is the error path only; the rest model is a pure function of the
                // coefficients, so the rebuild is deterministic.
                if let Ok(rest) = rest_model(&self.model.data, &self.model.rig, &self.coefficients)
                {
                    self.output = rest;
                }
                Err(e)
            }
        }
    }
}

type Scalar = f64;
include!("kernels/coefficients.rs");
include!("kernels/evaluation.rs");
