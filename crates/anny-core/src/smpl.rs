// Native forward adapter for explicitly supplied SMPL / SMPL-X model data.
// Port of NAVER Anny, Copyright (C) 2025 NAVER Corp. Apache-2.0.
// Model assets are separately licensed and are NEVER bundled or downloaded here.
use crate::{
    config::{BoneOrientation, RigConfig},
    ensure,
    math::*,
    model::{self, ModelData, ModelOutput},
    tensor::Archive,
    Error, PoseParameterization, Result, SkinningMethod, Tensor,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SmplKind {
    Smpl,
    Smplx,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SmplConfig {
    pub kind: SmplKind,
    pub num_betas: usize,
    pub num_expression_coeffs: usize,
    pub pose_corrective: bool,
    pub use_pca: bool,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SmplParameters {
    pub betas: Value,
    pub expression: Value,
    pub global_orient: Value,
    pub transl: Value,
    pub body_pose: Value,
    pub leye_pose: Value,
    pub reye_pose: Value,
    pub left_hand_pose: Value,
    pub right_hand_pose: Value,
    pub jaw_pose: Value,
}
#[derive(Clone, Debug)]
pub struct SmplModel {
    pub data: ModelData,
    pub config: SmplConfig,
    pub pose_mean: Tensor,
    pub left_hand_components: Option<Tensor>,
    pub right_hand_components: Option<Tensor>,
}
impl SmplModel {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut a = Archive::from_bytes(bytes)?;
        let config: SmplConfig =
            serde_json::from_str(a.metadata.get("anny_smpl_config").ok_or_else(|| {
                Error::Invalid("missing anny_smpl_config; use tools/export_smpl.py".into())
            })?)?;
        let pose_mean = a.tensors.remove("smpl_pose_mean");
        let left = a.tensors.remove("smpl_left_hand_components");
        let right = a.tensors.remove("smpl_right_hand_components");
        let data = ModelData::from_archive(a)?;
        let j = data.bone_count();
        let model = Self {
            data,
            config,
            pose_mean: pose_mean.unwrap_or_else(|| Tensor::zeros(vec![1, j, 3])),
            left_hand_components: left,
            right_hand_components: right,
        };
        model.validate()?;
        Ok(model)
    }
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_bytes(&std::fs::read(path)?)
    }
    pub fn validate(&self) -> Result<()> {
        self.data.validate()?;
        let j = self.data.bone_count();
        ensure(
            j == if self.config.kind == SmplKind::Smpl {
                24
            } else {
                55
            },
            "SMPL expects 24 joints; SMPL-X expects 55",
        )?;
        self.pose_mean.expect_shape(&[1, j, 3], "SMPL pose mean")?;
        let c = self.config.num_betas
            + self.config.num_expression_coeffs
            + if self.config.pose_corrective {
                (j - 1) * 9
            } else {
                0
            };
        ensure(
            c == self.data.blendshape_count(),
            "SMPL basis count does not match configuration",
        )?;
        ensure(
            self.config.kind == SmplKind::Smplx
                || (!self.config.use_pca && self.config.num_expression_coeffs == 0),
            "SMPL cannot have expressions or hand PCA",
        )?;
        if self.config.use_pca {
            for t in [&self.left_hand_components, &self.right_hand_components] {
                let t = t
                    .as_ref()
                    .ok_or_else(|| Error::Invalid("missing hand PCA components".into()))?;
                ensure(
                    t.shape.len() == 2 && t.shape[1] == 45 && t.shape[0] > 0,
                    "hand components must be [P,45]",
                )?;
                t.validate()?;
            }
        }
        Ok(())
    }
    pub fn forward(&self, p: &SmplParameters) -> Result<ModelOutput> {
        self.validate()?;
        let j = self.data.bone_count();
        let is_x = self.config.kind == SmplKind::Smplx;
        let betas = values(&p.betas, self.config.num_betas, "betas")?;
        let expression = values(
            &p.expression,
            self.config.num_expression_coeffs,
            "expression",
        )?;
        let root = values(&p.global_orient, 3, "global_orient")?;
        let tr = values(&p.transl, 3, "transl")?;
        let body = values(&p.body_pose, if is_x { 63 } else { 69 }, "body_pose")?;
        let mut pieces = vec![root, body];
        if is_x {
            pieces.push(values(&p.jaw_pose, 3, "jaw_pose")?);
            pieces.push(values(&p.leye_pose, 3, "leye_pose")?);
            pieces.push(values(&p.reye_pose, 3, "reye_pose")?);
            for (v, c, name) in [
                (
                    &p.left_hand_pose,
                    &self.left_hand_components,
                    "left_hand_pose",
                ),
                (
                    &p.right_hand_pose,
                    &self.right_hand_components,
                    "right_hand_pose",
                ),
            ] {
                if self.config.use_pca {
                    let c = c.as_ref().unwrap();
                    let input = values(v, c.shape[0], name)?;
                    let mut out = Tensor::zeros(vec![input.shape[0], 45]);
                    for bi in 0..input.shape[0] {
                        for k in 0..c.shape[0] {
                            for i in 0..45 {
                                out.data[bi * 45 + i] +=
                                    input.data[bi * c.shape[0] + k] * c.data[k * 45 + i];
                            }
                        }
                    }
                    pieces.push(out);
                } else {
                    pieces.push(values(v, 45, name)?);
                }
            }
        }
        let mut sizes = pieces.iter().map(|v| v.shape[0]).collect::<Vec<_>>();
        sizes.extend([betas.shape[0], expression.shape[0], tr.shape[0]]);
        let b = model::broadcast(&sizes)?;
        let mut pose = model::identity_poses(b, j);
        let c = self.data.blendshape_count();
        let mut coefficients = Tensor::zeros(vec![b, c]);
        for bi in 0..b {
            let mut vectors = Vec::with_capacity(j * 3);
            for piece in &pieces {
                let n = piece.shape[1];
                vectors.extend_from_slice(
                    &piece.data[(bi % piece.shape[0]) * n..(bi % piece.shape[0] + 1) * n],
                );
            }
            for i in 0..j {
                let v =
                    vec3(&vectors[i * 3..i * 3 + 3]) + vec3(&self.pose_mean.data[i * 3..i * 3 + 3]);
                let r = rotvec(&v);
                let t = if i == 0 {
                    vec3(&tr.data[(bi % tr.shape[0]) * 3..(bi % tr.shape[0] + 1) * 3])
                } else {
                    Vec3::zeros()
                };
                write4(
                    &rigid(&r, &t),
                    &mut pose.data[(bi * j + i) * 16..(bi * j + i + 1) * 16],
                );
            }
            let mut offset = 0;
            for t in [&betas, &expression] {
                let n = t.shape[1];
                coefficients.data[bi * c + offset..bi * c + offset + n]
                    .copy_from_slice(&t.data[(bi % t.shape[0]) * n..(bi % t.shape[0] + 1) * n]);
                offset += n;
            }
            if self.config.pose_corrective {
                for i in 1..j {
                    let r = rotation(&mat4(&pose.data[(bi * j + i) * 16..(bi * j + i + 1) * 16]))
                        - Mat3::identity();
                    write3(
                        &r,
                        &mut coefficients.data
                            [bi * c + offset + (i - 1) * 9..bi * c + offset + i * 9],
                    );
                }
            }
        }
        let mut rig = RigConfig::parse("makehuman")?;
        rig.bone_orientation = BoneOrientation::Blender;
        rig.root_identity_orientation = false;
        model::forward_model(
            &self.data,
            &rig,
            &coefficients,
            &pose.nested_json(),
            PoseParameterization::LocalBoneWorld,
            SkinningMethod::Lbs,
            false,
        )
    }
}
fn values(value: &Value, n: usize, name: &str) -> Result<Tensor> {
    if value.is_null() {
        return Ok(Tensor::zeros(vec![1, n]));
    }
    let mut t = Tensor::from_nested(value)?;
    ensure(
        n > 0 || t.data.is_empty(),
        format!("{name} not supported by this model"),
    )?;
    if t.shape == [n] {
        t.shape = vec![1, n];
    } else if t.shape.len() >= 2 && t.shape[1..].iter().product::<usize>() == n {
        t.shape = vec![t.shape[0], n];
    }
    ensure(
        t.shape.len() == 2 && t.shape[0] > 0 && t.shape[1] == n,
        format!("{name} expects [B,{n}]"),
    )?;
    Ok(t)
}
