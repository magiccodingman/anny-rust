//! Native Anny pose clips, interpolation and skeletal animation export.
//! AMASS SMPL-X arrays can be decoded separately; their pose vectors are NEVER
//! silently treated as Anny bone rotations. Optional source models stay external.
use crate::{
    ensure,
    math::*,
    model,
    numpy::{self, NpyValue},
    scene::{Animation, CharacterExport, Scene},
    smpl::{SmplKind, SmplModel, SmplParameters},
    Anny, Error, Parameters, PoseParameterization, Result, Tensor,
};
use nalgebra::UnitQuaternion;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    #[default]
    F64,
    F32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PoseFrame {
    pub time: f64,
    pub pose_parameters: Value,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PoseClip {
    pub name: String,
    /// Static shape/expression and pose convention. Per-frame poses live below.
    pub parameters: Parameters,
    pub frames: Vec<PoseFrame>,
}
impl Default for PoseClip {
    fn default() -> Self {
        Self {
            name: "Anny motion".into(),
            parameters: Parameters::default(),
            frames: vec![],
        }
    }
}
fn invalid(s: &str) -> Error {
    Error::Invalid(s.into())
}
fn proper(h: &Mat4) -> Result<()> {
    checked_rigid(h)?;
    let r = rotation(h);
    ensure(
        (r.transpose() * r - Mat3::identity()).norm() < 1e-4 && (r.determinant() - 1.).abs() < 1e-4,
        "motion interpolation requires rigid rotations (no scale, shear or reflection)",
    )
}
impl PoseClip {
    pub fn validate(&self, model: &Anny) -> Result<()> {
        ensure(
            !self.frames.is_empty() && self.frames.len() <= 100_000,
            "motion needs 1..=100000 frames",
        )?;
        ensure(
            self.parameters.pose_parameters.is_null(),
            "clip base parameters must not also contain a pose",
        )?;
        ensure(
            model.coefficients(&self.parameters)?.shape[0] == 1,
            "a motion clip has one fixed character shape",
        )?;
        let mut previous = None;
        for frame in &self.frames {
            ensure(
                frame.time.is_finite() && frame.time >= 0. && (frame.time as f32).is_finite(),
                "invalid motion time",
            )?;
            if let Some(p) = previous {
                ensure(
                    (frame.time as f32) > p,
                    "motion times collapse or are not increasing in f32",
                )?;
            }
            previous = Some(frame.time as f32);
            let pose = model::parse_pose(&frame.pose_parameters, &model.data.metadata.bone_labels)?;
            ensure(
                pose.shape[0] == 1,
                "each clip frame must contain one pose, not a batch",
            )?;
            for row in pose.data.chunks_exact(16) {
                proper(&mat4(row))?;
            }
        }
        Ok(())
    }
    pub fn frame_parameters(&self, index: usize) -> Result<Parameters> {
        let frame = self
            .frames
            .get(index)
            .ok_or_else(|| invalid("motion frame outside clip"))?;
        let mut p = self.parameters.clone();
        p.pose_parameters = frame.pose_parameters.clone();
        Ok(p)
    }
    /// Axis-angle vectors are radians and use the explicitly selected Anny mode.
    pub fn from_rotvecs(
        model: &Anny,
        vectors: &Tensor,
        translations: &Tensor,
        fps: f64,
        mode: PoseParameterization,
    ) -> Result<Self> {
        ensure(
            vectors.shape.len() == 3
                && vectors.shape[0] > 0
                && vectors.shape[1..] == [model.data.bone_count(), 3],
            "rotations must be [frames,bones,3]",
        )?;
        let count = vectors.shape[0];
        ensure(
            count <= 100_000 && fps.is_finite() && fps > 0.,
            "invalid frame rate/count",
        )?;
        vectors.validate()?;
        translations.expect_shape(&[count, 3], "root translations")?;
        let mut clip = Self {
            parameters: Parameters {
                pose_parameterization: Some(mode),
                ..Default::default()
            },
            ..Default::default()
        };
        let j = model.data.bone_count();
        ensure(
            model.data.metadata.bone_parents[0] < 0
                && model
                    .data
                    .metadata
                    .bone_parents
                    .iter()
                    .filter(|p| **p < 0)
                    .count()
                    == 1,
            "rotvec clip construction requires a single root at joint zero",
        )?;
        for frame in 0..count {
            let mut pose = model::identity_poses(1, j);
            for i in 0..j {
                let r = rotvec(&vec3(
                    &vectors.data[(frame * j + i) * 3..(frame * j + i + 1) * 3],
                ));
                let tr = if i == 0 {
                    vec3(&translations.data[frame * 3..frame * 3 + 3])
                } else {
                    Vec3::zeros()
                };
                write4(&rigid(&r, &tr), &mut pose.data[i * 16..i * 16 + 16]);
            }
            clip.frames.push(PoseFrame {
                time: frame as f64 / fps,
                pose_parameters: pose.nested_json(),
            });
        }
        clip.validate(model)?;
        Ok(clip)
    }
    /// Quaternion SLERP + translation interpolation in the selected pose convention.
    /// Both endpoints are retained, including when duration is not a multiple of 1/fps.
    pub fn resample(&self, model: &Anny, fps: f64) -> Result<Self> {
        self.validate(model)?;
        ensure(
            fps.is_finite() && fps > 0.,
            "fps must be finite and positive",
        )?;
        let first = self.frames[0].time;
        let last = self.frames.last().unwrap().time;
        let steps = ((last - first) * fps).floor();
        ensure(
            steps.is_finite() && steps < 100_000.,
            "resampled clip exceeds frame limit",
        )?;
        let mut times: Vec<_> = (0..=steps as usize)
            .map(|i| first + i as f64 / fps)
            .collect();
        if (times.last().copied().unwrap() as f32) < last as f32 {
            times.push(last);
        } else {
            *times.last_mut().unwrap() = last;
        }
        let poses = self
            .frames
            .iter()
            .map(|f| model::parse_pose(&f.pose_parameters, &model.data.metadata.bone_labels))
            .collect::<Result<Vec<_>>>()?;
        let mut output = Self {
            frames: Vec::new(),
            ..self.clone()
        };
        for time in times {
            let hi = self
                .frames
                .partition_point(|f| f.time < time)
                .min(self.frames.len() - 1);
            let lo = hi.saturating_sub(1);
            let alpha = if hi == lo {
                0.
            } else {
                ((time - self.frames[lo].time) / (self.frames[hi].time - self.frames[lo].time))
                    .clamp(0., 1.)
            };
            let mut result = model::identity_poses(1, model.data.bone_count());
            for (i, row) in result.data.chunks_exact_mut(16).enumerate() {
                let a = mat54T4 1ND2<°