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
        ensure(
            mask.shape.len() == 2 && mask.shape[1] == 26,
            "phenotype mask must have 26 columns",
        )?;
        ensure(
            mask.data.iter().all(|&x| x == 0. || x == 1.),
            "phenotype mask must contain zero or one",
        )?;
        let local_change_labels = data.metadata.local_change_labels();
        let facial_action_labels = data.metadata.facial_action_labels();
        ensure(
            mask.shape[0] + local_change_labels.len() * 2 + facial_action_labels.len()
                == data.blendshape_count(),
            "blendshape blocks do not match metadata",
        )?;
        // Coefficient packing is part of the compatibility contract, not map order.
        let m = mask.shape[0];
        let f = facial_action_labels.len();
        ensure(
            data.metadata.blendshape_labels[..m]
                .iter()
                .all(|s| !s.starts_with("local_change:") && !s.starts_with("facial_action:"))
                && data.metadata.blendshape_labels[m..m + f]
                    .iter()
                    .all(|s| s.starts_with("facial_action:"))
                && data.metadata.blendshape_labels[m + f..]
                    .iter()
                    .all(|s| s.starts_with("local_change:")),
            "blendshape blocks must be macro, facial, then paired local",
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
}

type Scalar = f64;
include!("kernels/coefficients.rs");
include!("kernels/evaluation.rs");
