//! Native single-precision evaluation using the same source kernels as f64.
//! Conversion/loading happens once; coefficients, SVD, FK and skinning execute
//! on f32 buffers. This is not an f64 evaluation followed by an output cast.
use crate::{
    config::*,
    ensure,
    model::{ModelMetadata, Parameters},
    tensor::Kind,
    Error, Result,
};
use safetensors::{tensor::TensorView, Dtype, SafeTensors};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};

/// Owned f32 tensor. Discrete fields are checked for exact representability.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorF32 {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    #[serde(default)]
    pub kind: Kind,
}
impl TensorF32 {
    pub fn new(shape: impl Into<Vec<usize>>, data: Vec<f32>) -> Result<Self> {
        let result = Self {
            shape: shape.into(),
            data,
            kind: Kind::Float,
        };
        result.validate()?;
        Ok(result)
    }
    pub fn zeros(shape: impl Into<Vec<usize>>) -> Self {
        let shape = shape.into();
        let n = shape.iter().product();
        Self {
            shape,
            data: vec![0.; n],
            kind: Kind::Float,
        }
    }
    /// Construct an exactly represented discrete tensor; never round an index.
    pub fn indices(shape: impl Into<Vec<usize>>, data: Vec<usize>) -> Result<Self> {
        ensure(
            data.iter().all(|&x| x <= 16_777_216),
            "index exceeds the f32 exact-integer limit",
        )?;
        let result = Self {
            shape: shape.into(),
            data: data.into_iter().map(|x| x as f32).collect(),
            kind: Kind::Index,
        };
        result.validate()?;
        Ok(result)
    }
    pub fn validate(&self) -> Result<()> {
        let n = self
            .shape
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .ok_or_else(|| Error::Invalid("tensor shape overflow".into()))?;
        ensure(
            n == self.data.len(),
            format!(
                "shape {:?} needs {n} entries, got {}",
                self.shape,
                self.data.len()
            ),
        )?;
        ensure(
            self.data.iter().all(|x| x.is_finite()),
            "non-finite tensor value",
        )?;
        if self.kind == Kind::Index {
            ensure(
                self.data
                    .iter()
                    .all(|x| x.fract() == 0. && x.abs() <= 16_777_216.),
                "integer outside exact supported range",
            )?;
        }
        if self.kind == Kind::Bool {
            ensure(
                self.data.iter().all(|x| *x == 0. || *x == 1.),
                "invalid boolean",
            )?;
        }
        Ok(())
    }
    /// Check that this tensor has exactly `shape`, requiring an O(1) structural check only.
    ///
    /// See [`crate::tensor::Tensor::expect_shape`] for why this is not a full validation: the
    /// finiteness scan is O(elements) and this sits on the evaluation hot path for tensors in the
    /// hundreds of megabytes, which cost ~8 ms per call on the typed path. Finiteness is enforced
    /// once, at construction; debug builds keep the full scan.
    pub fn expect_shape(&self, shape: &[usize], name: &str) -> Result<()> {
        #[cfg(debug_assertions)]
        self.validate()?;
        self.checked_shape(shape, name)
    }
    /// The O(1) half of [`TensorF32::expect_shape`]: rank/product/entry-count consistency plus the
    /// expected shape.
    pub fn checked_shape(&self, shape: &[usize], name: &str) -> Result<()> {
        let n = self
            .shape
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .ok_or_else(|| Error::Invalid("tensor shape overflow".into()))?;
        ensure(
            n == self.data.len(),
            format!(
                "shape {:?} needs {n} entries, got {}",
                self.shape,
                self.data.len()
            ),
        )?;
        ensure(
            self.shape == shape,
            format!("{name}: expected {shape:?}, got {:?}", self.shape),
        )
    }
    pub fn checked_indices(&self, bound: usize, name: &str) -> Result<Vec<usize>> {
        self.validate()?;
        // See `crate::tensor::Tensor::checked_indices`: building the message eagerly costs one
        // allocation per element over mesh-size arrays.
        let mut out = Vec::with_capacity(self.data.len());
        for &x in &self.data {
            if !(x >= 0. && x.fract() == 0. && (x as usize) < bound) {
                return Err(Error::Invalid(format!(
                    "{name}: index {x} outside [0,{bound})"
                )));
            }
            out.push(x as usize);
        }
        Ok(out)
    }
    pub fn select(&self, axis: usize, indices: &[usize]) -> Result<Self> {
        ensure(
            axis < self.shape.len(),
            "selection axis outside tensor rank",
        )?;
        ensure(
            indices.iter().all(|&i| i < self.shape[axis]),
            "selection index outside tensor",
        )?;
        let inner: usize = self.shape[axis + 1..].iter().product();
        let outer: usize = self.shape[..axis].iter().product();
        let mut shape = self.shape.clone();
        shape[axis] = indices.len();
        let mut data = Vec::with_capacity(outer * indices.len() * inner);
        for a in 0..outer {
            for &i in indices {
                let start = (a * self.shape[axis] + i) * inner;
                data.extend_from_slice(&self.data[start..start + inner]);
            }
        }
        Ok(Self {
            shape,
            data,
            kind: self.kind,
        })
    }
    pub fn nested_json(&self) -> Value {
        fn build(shape: &[usize], data: &[f32], kind: Kind) -> Value {
            if shape.is_empty() {
                return if kind == Kind::Bool {
                    Value::Bool(data[0] != 0.)
                } else if kind == Kind::Index {
                    Value::from(data[0] as i64)
                } else {
                    Value::from(data[0])
                };
            }
            let step: usize = shape[1..].iter().product();
            Value::Array(
                (0..shape[0])
                    .map(|i| build(&shape[1..], &data[i * step..(i + 1) * step], kind))
                    .collect(),
            )
        }
        build(&self.shape, &self.data, self.kind)
    }
    pub fn from_nested(value: &Value) -> Result<Self> {
        fn dimensions(v: &Value) -> Vec<usize> {
            if let Some(a) = v.as_array() {
                let mut s = vec![a.len()];
                if let Some(first) = a.first() {
                    s.extend(dimensions(first));
                }
                s
            } else {
                vec![]
            }
        }
        fn visit(v: &Value, shape: &[usize], data: &mut Vec<f32>) -> Result<()> {
            if shape.is_empty() {
                data.push(
                    v.as_f64()
                        .ok_or_else(|| Error::Invalid("tensor entries must be numbers".into()))?
                        as f32,
                );
            } else {
                let a = v
                    .as_array()
                    .ok_or_else(|| Error::Invalid("mixed tensor ranks".into()))?;
                ensure(a.len() == shape[0], "ragged tensor input")?;
                for x in a {
                    visit(x, &shape[1..], data)?;
                }
            }
            Ok(())
        }
        let shape = dimensions(value);
        let mut data = vec![];
        visit(value, &shape, &mut data)?;
        Self::new(shape, data)
    }
}

impl TensorF32 {
    pub fn from_reference(t: &crate::Tensor) -> Result<Self> {
        t.validate()?;
        let data: Vec<f32> = t.data.iter().map(|&x| x as f32).collect();
        ensure(data.iter().all(|x| x.is_finite()), "value overflows f32")?;
        if t.kind != Kind::Float {
            ensure(
                data.iter().zip(&t.data).all(|(&a, &b)| a as f64 == b),
                "discrete field is not exactly representable in f32",
            )?;
        }
        let result = Self {
            shape: t.shape.clone(),
            data,
            kind: t.kind,
        };
        result.validate()?;
        Ok(result)
    }
    /// Widen an already evaluated result for legacy authoring/serialization APIs.
    /// This conversion performs no model evaluation.
    pub fn to_reference(&self) -> crate::Tensor {
        crate::Tensor {
            shape: self.shape.clone(),
            data: self.data.iter().map(|&x| x as f64).collect(),
            kind: self.kind,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ModelDataF32 {
    pub metadata: ModelMetadata,
    pub arrays: BTreeMap<String, TensorF32>,
}
impl ModelDataF32 {
    pub fn get(&self, name: &str) -> Result<&TensorF32> {
        self.arrays
            .get(name)
            .ok_or_else(|| Error::Invalid(format!("ModelData is missing {name}")))
    }
    pub fn vertex_count(&self) -> usize {
        self.arrays["template_vertices"].shape[0]
    }
    pub fn bone_count(&self) -> usize {
        self.metadata.bone_labels.len()
    }
    pub fn blendshape_count(&self) -> usize {
        self.metadata.blendshape_labels.len()
    }
}
#[derive(Clone, Debug, Default)]
pub struct ModelOutputF32 {
    pub arrays: BTreeMap<String, TensorF32>,
}
impl ModelOutputF32 {
    pub fn get(&self, name: &str) -> Result<&TensorF32> {
        self.arrays
            .get(name)
            .ok_or_else(|| Error::Invalid(format!("missing output {name}")))
    }
    pub fn to_json(&self) -> Value {
        Value::Object(
            self.arrays
                .iter()
                .map(|(k, t)| (k.clone(), t.nested_json()))
                .collect(),
        )
    }
    pub fn to_reference(&self) -> crate::ModelOutput {
        crate::ModelOutput {
            arrays: self
                .arrays
                .iter()
                .map(|(k, t)| (k.clone(), t.to_reference()))
                .collect(),
        }
    }
    /// F32 attributes are stored as F32, not widened in the output archive.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serialize(&self.arrays, HashMap::new())
    }
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        std::fs::write(path, self.to_bytes()?)?;
        Ok(())
    }
}

/// A safetensors payload decoded straight into f32 storage, with no f64 intermediate.
pub struct ArchiveF32 {
    pub tensors: BTreeMap<String, TensorF32>,
    pub metadata: HashMap<String, String>,
}

impl ArchiveF32 {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let (_, info) = SafeTensors::read_metadata(bytes)?;
        let metadata = info.metadata().clone().unwrap_or_default();
        let s = SafeTensors::deserialize(bytes)?;
        let mut tensors = BTreeMap::new();
        for (name, v) in s.iter() {
            let (kind, data) = crate::tensor::decode_f32(v.dtype(), v.data())?;
            let t = TensorF32 {
                shape: v.shape().to_vec(),
                data,
                kind,
            };
            t.validate()?;
            tensors.insert(name.into(), t);
        }
        Ok(Self { tensors, metadata })
    }
}

/// A reusable f32 runtime. Static model arrays are converted once at creation.
/// The original f64 model remains valid and unchanged.
#[derive(Clone, Debug)]
pub struct AnnyF32 {
    data: ModelDataF32,
    config: AnnyConfig,
    rig: RigConfig,
    phenotype_labels: Vec<String>,
    local_change_labels: Vec<String>,
    facial_action_labels: Vec<String>,
}
impl AnnyF32 {
    pub fn from_anny(source: &crate::Anny) -> Result<Self> {
        source.data.validate()?;
        source.config.validate()?;
        let rig = source.resolved_rig().clone();
        let data = ModelDataF32 {
            metadata: source.data.metadata.clone(),
            arrays: source
                .data
                .arrays
                .iter()
                .map(|(k, t)| Ok((k.clone(), TensorF32::from_reference(t)?)))
                .collect::<Result<_>>()?,
        };
        validate_orientation(&data, rig.bone_orientation)?;
        Ok(Self {
            data,
            config: source.config.clone(),
            rig,
            phenotype_labels: source.phenotype_labels.clone(),
            local_change_labels: source.local_change_labels.clone(),
            facial_action_labels: source.facial_action_labels.clone(),
        })
    }
    pub fn from_model_data(data: crate::ModelData, config: AnnyConfig) -> Result<Self> {
        Self::from_anny(&crate::Anny::from_model_data(data, config)?)
    }
    pub fn from_bytes(bytes: &[u8], config: Option<AnnyConfig>) -> Result<Self> {
        // Decode straight into f32 storage. Going through `Anny::from_bytes` widened the payload to
        // 205.9 MB of f64 and converted every element back down, which measured 96 ms of the 229 ms
        // reload; the payload is already f32, so read it as f32 and run the same checks.
        let a = ArchiveF32::from_bytes(bytes)?;
        let version = a
            .metadata
            .get("data_version")
            .and_then(|s| s.parse::<usize>().ok());
        ensure(
            version == Some(crate::DATA_VERSION),
            format!(
                "ModelData data_version {version:?}; expected {}",
                crate::DATA_VERSION
            ),
        )?;
        let header = a
            .metadata
            .get("metadata")
            .ok_or_else(|| Error::Invalid("ModelData missing metadata header".into()))?;
        let metadata: crate::model::ModelMetadata = serde_json::from_str(header)?;
        let config = match config {
            Some(c) => c,
            None => a
                .metadata
                .get("anny_rust_config")
                .map(|v| serde_json::from_str(v))
                .transpose()?
                .unwrap_or_default(),
        };
        config.validate()?;
        let rig = config.rig.resolve()?;
        let data = ModelDataF32 {
            metadata,
            arrays: a.tensors,
        };
        let mask = data.get("stacked_phenotype_blend_shapes_mask")?;
        let (local_change_labels, facial_action_labels) = crate::model::validate_model_blocks(
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
    pub fn load(path: impl AsRef<std::path::Path>, config: Option<AnnyConfig>) -> Result<Self> {
        Self::from_bytes(&std::fs::read(path)?, config)
    }
    pub fn data(&self) -> &ModelDataF32 {
        &self.data
    }
    pub fn config(&self) -> &AnnyConfig {
        &self.config
    }
    pub fn describe(&self) -> Value {
        json!({"dtype":"f32", "vertices":self.data.vertex_count(), "bones":self.data.bone_count(),
            "faces":self.data.arrays["faces"].shape[0], "blendshapes":self.data.blendshape_count(),
            "bone_labels":self.data.metadata.bone_labels, "bone_parents":self.data.metadata.bone_parents,
            "phenotype_labels":self.phenotype_labels, "local_change_labels":self.local_change_labels,
            "facial_action_labels":self.facial_action_labels, "config":self.config})
    }
    pub fn rest_model(&self, coefficients: &TensorF32) -> Result<ModelOutputF32> {
        rest_model(&self.data, &self.rig, coefficients)
    }
    pub fn forward(&self, p: &Parameters) -> Result<ModelOutputF32> {
        forward_model(
            &self.data,
            &self.rig,
            &self.coefficients(p)?,
            &p.pose_parameters,
            p.pose_parameterization
                .unwrap_or(self.config.pose_parameterization),
            self.config.skinning_method,
            p.return_bone_ends,
        )
    }
    pub fn pose_parameters(
        &self,
        output: &ModelOutputF32,
        mode: PoseParameterization,
    ) -> Result<TensorF32> {
        pose_parameters(&self.data, output, mode)
    }
    /// Build a reusable pose-only evaluation context for a fixed parameter set.
    ///
    /// The f32 mirror of [`crate::Anny::pose_session`], and the one that matters for games: the typed
    /// path is what a Unity or WASM runtime evaluates, so without this the product surfaces would
    /// still rebuild the whole rest model on every pose change. See that method for the semantics.
    pub fn pose_session(&self, p: &Parameters) -> Result<PoseSessionF32<'_>> {
        let coefficients = self.coefficients(p)?;
        let output = rest_model(&self.data, &self.rig, &coefficients)?;
        Ok(PoseSessionF32 {
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
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serialize(
            &self.data.arrays,
            HashMap::from([
                ("data_version".into(), crate::DATA_VERSION.to_string()),
                (
                    "metadata".into(),
                    serde_json::to_string(&self.data.metadata)?,
                ),
                ("metadata_class".into(), "ModelMetadata".into()),
                (
                    "anny_rust_config".into(),
                    serde_json::to_string(&self.config)?,
                ),
                ("anny_rust_upstream".into(), crate::UPSTREAM_REVISION.into()),
                ("anny_rust_dtype".into(), "f32".into()),
            ]),
        )
    }
}
impl crate::Anny {
    pub fn to_f32(&self) -> Result<AnnyF32> {
        AnnyF32::from_anny(self)
    }
}

/// A reusable pose-only evaluation context for the typed f32 path; see
/// [`AnnyF32::pose_session`].
///
/// [`PoseSessionF32::update`] returns the same [`ModelOutputF32`] shape [`AnnyF32::forward`] returns and
/// is bit-identical to the corresponding `forward` call for the same parameters.
pub struct PoseSessionF32<'a> {
    model: &'a AnnyF32,
    coefficients: TensorF32,
    output: ModelOutputF32,
    pose_parameterization: PoseParameterization,
    skinning: SkinningMethod,
    return_bone_ends: bool,
}
impl PoseSessionF32<'_> {
    /// The phenotype/local-change/facial coefficients this session was built with.
    pub fn coefficients(&self) -> &TensorF32 {
        &self.coefficients
    }
    /// Evaluate a pose against the cached rest model, reusing the previous output buffers.
    pub fn update(&mut self, pose: &Value) -> Result<&ModelOutputF32> {
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
                // Same reasoning as the f64 session: the failed call consumed the output the rest
                // arrays live in, so the session rebuilds them to stay usable after a bad pose.
                if let Ok(rest) = rest_model(&self.model.data, &self.model.rig, &self.coefficients)
                {
                    self.output = rest;
                }
                Err(e)
            }
        }
    }
}
fn serialize(
    arrays: &BTreeMap<String, TensorF32>,
    metadata: HashMap<String, String>,
) -> Result<Vec<u8>> {
    let mut buffers = Vec::new();
    for (name, t) in arrays {
        t.validate()?;
        let (dtype, bytes) = match t.kind {
            Kind::Float => (
                Dtype::F32,
                t.data
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
            Kind::Index => (
                Dtype::I64,
                t.data
                    .iter()
                    .flat_map(|x| (*x as i64).to_le_bytes())
                    .collect(),
            ),
            Kind::Bool => (
                Dtype::BOOL,
                t.data.iter().map(|x| u8::from(*x != 0.)).collect(),
            ),
        };
        buffers.push((name, t.shape.clone(), dtype, bytes));
    }
    let views = buffers
        .iter()
        .map(|(n, s, d, b)| Ok(((*n).clone(), TensorView::new(*d, s.clone(), b)?)))
        .collect::<Result<Vec<_>>>()?;
    Ok(safetensors::serialize(views, &Some(metadata))?)
}

pub mod math {
    type Scalar = f32;
    include!("kernels/math.rs");
}
use math::*;
type Scalar = f32;
type Tensor = TensorF32;
type ModelData = ModelDataF32;
type ModelOutput = ModelOutputF32;
type Anny = AnnyF32;
include!("kernels/coefficients.rs");
include!("kernels/evaluation.rs");
