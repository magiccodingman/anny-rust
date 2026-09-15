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
                let a = mat4(&poses[lo].data[i * 16..i * 16 + 16]);
                let b = mat4(&poses[hi].data[i * 16..i * 16 + 16]);
                let qa = UnitQuaternion::from_matrix(&rotation(&a));
                let qb = UnitQuaternion::from_matrix(&rotation(&b));
                let q = qa.slerp(&qb, alpha);
                let tr = translation(&a) * (1. - alpha) + translation(&b) * alpha;
                write4(&rigid(q.to_rotation_matrix().matrix(), &tr), row);
            }
            output.frames.push(PoseFrame {
                time,
                pose_parameters: result.nested_json(),
            });
        }
        output.validate(model)?;
        Ok(output)
    }
    /// Same-geometry rig transfer using the library's checked name/rest mapping.
    pub fn transfer(
        &self,
        source: &Anny,
        target: &Anny,
        mode: PoseParameterization,
    ) -> Result<Self> {
        self.validate(source)?;
        let mut clip = Self {
            name: self.name.clone(),
            parameters: Parameters {
                pose_parameterization: Some(mode),
                ..self.parameters.clone()
            },
            frames: Vec::new(),
        };
        for (i, frame) in self.frames.iter().enumerate() {
            let pose = crate::tools::transfer_pose_parameters(
                source,
                target,
                &self.frame_parameters(i)?,
                mode,
            )?;
            clip.frames.push(PoseFrame {
                time: frame.time,
                pose_parameters: pose.nested_json(),
            });
        }
        clip.validate(target)?;
        Ok(clip)
    }
    pub fn to_scene(&self, model: &Anny, precision: Precision) -> Result<Scene> {
        self.validate(model)?;
        let single = if precision == Precision::F32 {
            Some(model.to_f32()?)
        } else {
            None
        };
        let evaluate = |p: &Parameters| match &single {
            Some(m) => Ok(m.forward(p)?.to_reference()),
            None => model.forward(p),
        };
        let first = self.frame_parameters(0)?;
        let output = evaluate(&first)?;
        let mut scene = Scene::new();
        let object = scene.add_evaluated_character(
            model,
            &first,
            &CharacterExport {
                name: self.name.clone(),
                rigged: true,
                ..Default::default()
            },
            &output,
        )?;
        let bones_of = |output: &crate::ModelOutput| -> Result<Vec<_>> {
            Ok(output
                .get("bone_poses")?
                .data
                .chunks_exact(16)
                .map(mat4)
                .collect())
        };
        // A clip is one fixed character shape with one pose per frame, which is what a pose session is
        // for: its coefficients and rest model are evaluated once, and each frame then pays only for its
        // own pose. Frame 0 keeps the full `forward` above because the scene needs its whole output.
        let mut poses = vec![bones_of(&output)?];
        match &single {
            Some(m) => {
                let mut session = m.pose_session(&first)?;
                for i in 1..self.frames.len() {
                    let frame = self.frame_parameters(i)?;
                    poses.push(bones_of(
                        &session.update(&frame.pose_parameters)?.to_reference(),
                    )?);
                }
            }
            None => {
                let mut session = model.pose_session(&first)?;
                for i in 1..self.frames.len() {
                    let frame = self.frame_parameters(i)?;
                    poses.push(bones_of(session.update(&frame.pose_parameters)?)?);
                }
            }
        }
        scene.add_animation(Animation {
            name: self.name.clone(),
            object,
            times: self.frames.iter().map(|f| f.time).collect(),
            bone_poses: poses,
        })?;
        Ok(scene)
    }
}

/// AMASS SMPL-X sequence layout used by upstream's amass_to_anny.py.
/// Decoding does not need SMPL-X assets. Evaluating/fitting does, explicitly.
#[derive(Clone, Debug)]
pub struct AmassSequence {
    pub betas: Tensor,
    pub root_orient: Tensor,
    pub pose_body: Tensor,
    pub pose_hand: Tensor,
    pub pose_jaw: Tensor,
    pub translations: Tensor,
    pub fps: Option<f64>,
    pub gender: Option<String>,
}
impl AmassSequence {
    pub fn from_npz(bytes: &[u8]) -> Result<Self> {
        Self::from_arrays(&numpy::read_npz(bytes, numpy::Limits::default())?)
    }
    pub fn from_arrays(arrays: &BTreeMap<String, NpyValue>) -> Result<Self> {
        let get = |name: &str| {
            arrays
                .get(name)
                .ok_or_else(|| {
                    Error::Invalid(format!(
                        "AMASS SMPL-X missing {name}; packed SMPL-H poses are not interchangeable"
                    ))
                })?
                .numeric()
                .cloned()
        };
        let root = get("root_orient")?;
        root.validate()?;
        ensure(
            root.shape.len() == 2 && root.shape[0] > 0 && root.shape[1] == 3,
            "root_orient must be [frames,3]",
        )?;
        let count = root.shape[0];
        ensure(count <= 100_000, "AMASS frame limit exceeded")?;
        let body = get("pose_body")?;
        body.expect_shape(&[count, 63], "pose_body")?;
        let betas = get("betas")?;
        betas.validate()?;
        ensure(
            betas.shape.len() == 1 && !betas.data.is_empty(),
            "betas must be a nonempty vector",
        )?;
        let optional = |name: &str, n: usize| -> Result<Tensor> {
            let t = match arrays.get(name) {
                Some(v) => v.numeric()?.clone(),
                None => Tensor::zeros(vec![count, n]),
            };
            t.expect_shape(&[count, n], name)?;
            Ok(t)
        };
        let fps = arrays
            .get("mocap_frame_rate")
            .or_else(|| arrays.get("mocap_framerate"))
            .map(|v| -> Result<f64> {
                let t = v.numeric()?;
                ensure(
                    t.data.len() == 1 && t.data[0].is_finite() && t.data[0] > 0.,
                    "invalid AMASS frame rate",
                )?;
                Ok(t.data[0])
            })
            .transpose()?;
        let gender = arrays
            .get("gender")
            .map(|v| v.scalar_text().map(String::from))
            .transpose()?;
        Ok(Self {
            betas,
            root_orient: root,
            pose_body: body,
            pose_hand: optional("pose_hand", 90)?,
            pose_jaw: optional("pose_jaw", 3)?,
            translations: optional("trans", 3)?,
            fps,
            gender,
        })
    }
    pub fn frame_count(&self) -> usize {
        self.root_orient.shape.first().copied().unwrap_or(0)
    }
    pub fn validate(&self) -> Result<()> {
        let n = self.frame_count();
        ensure(n > 0 && n <= 100_000, "invalid AMASS frame count")?;
        self.root_orient.expect_shape(&[n, 3], "root_orient")?;
        self.pose_body.expect_shape(&[n, 63], "pose_body")?;
        self.pose_hand.expect_shape(&[n, 90], "pose_hand")?;
        self.pose_jaw.expect_shape(&[n, 3], "pose_jaw")?;
        self.translations.expect_shape(&[n, 3], "trans")?;
        self.betas.validate()?;
        ensure(
            self.betas.shape.len() == 1 && !self.betas.data.is_empty(),
            "invalid beta vector",
        )?;
        Ok(())
    }
    pub fn describe(&self) -> Value {
        json!({"layout":"AMASS SMPL-X","frames":self.frame_count(),"fps":self.fps,"gender":self.gender,"betas":self.betas.data.len(),"requires_external_model_to_evaluate":true})
    }
    /// Convert data fields, not skeleton semantics. Requires a compatible supplied
    /// SMPL-X model (axis-angle hands, not PCA). No files are downloaded.
    pub fn parameters(
        &self,
        frame: usize,
        source: &SmplModel,
        rest: bool,
    ) -> Result<SmplParameters> {
        self.validate()?;
        ensure(frame < self.frame_count(), "AMASS frame outside sequence")?;
        ensure(
            source.config.kind == SmplKind::Smplx && !source.config.use_pca,
            "AMASS adapter needs SMPL-X with non-PCA hand rotations",
        )?;
        ensure(
            source.config.num_betas <= self.betas.data.len(),
            "AMASS beta vector is too short for source model",
        )?;
        let row = |t: &Tensor, offset: usize, n: usize| {
            if rest {
                json!(vec![0.; n])
            } else {
                json!(&t.data[frame * t.shape[1] + offset..frame * t.shape[1] + offset + n])
            }
        };
        Ok(SmplParameters {
            betas: json!(&self.betas.data[..source.config.num_betas]),
            global_orient: row(&self.root_orient, 0, 3),
            body_pose: row(&self.pose_body, 0, 63),
            left_hand_pose: row(&self.pose_hand, 0, 45),
            right_hand_pose: row(&self.pose_hand, 45, 45),
            jaw_pose: row(&self.pose_jaw, 0, 3),
            transl: row(&self.translations, 0, 3),
            ..Default::default()
        })
    }
}

/// Explicit correspondence from Anny's vertex order to the supplied source mesh.
/// Identity is an assertion by the caller, not inferred from matching counts.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum VertexMap {
    Identity,
    Weighted {
        indices: Vec<Vec<usize>>,
        weights: Vec<Vec<f64>>,
    },
}
impl VertexMap {
    pub fn map(&self, source: &Tensor, count: usize) -> Result<Tensor> {
        source.validate()?;
        ensure(
            source.shape.len() == 3 && source.shape[0] == 1 && source.shape[2] == 3,
            "source frame must be [1,vertices,3]",
        )?;
        match self {
            Self::Identity => {
                ensure(
                    source.shape[1] == count,
                    "explicit identity correspondence vertex count mismatch",
                )?;
                Ok(source.clone())
            }
            Self::Weighted { indices, weights } => {
                ensure(
                    indices.len() == count && weights.len() == count,
                    "vertex correspondence row count mismatch",
                )?;
                let mut output = Tensor::zeros(vec![1, count, 3]);
                for (row, (ids, ws)) in indices.iter().zip(weights).enumerate() {
                    ensure(
                        !ids.is_empty() && ids.len() <= 64 && ids.len() == ws.len(),
                        "invalid correspondence row",
                    )?;
                    ensure(
                        ids.iter().all(|&i| i < source.shape[1])
                            && ws.iter().all(|w| w.is_finite() && *w >= 0.)
                            && (ws.iter().sum::<f64>() - 1.).abs() < 1e-6,
                        "invalid correspondence indices or normalized weights",
                    )?;
                    for (&id, &weight) in ids.iter().zip(ws) {
                        for k in 0..3 {
                            output.data[row * 3 + k] += source.data[id * 3 + k] * weight;
                        }
                    }
                }
                output.validate()?;
                Ok(output)
            }
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AmassFitOptions {
    pub frame_step: usize,
    pub max_frames: usize,
    pub fps_override: Option<f64>,
    pub setup: crate::inverter::InverterOptions,
    pub shape: crate::inverter::FitOptions,
    pub pose_iterations: usize,
}
impl Default for AmassFitOptions {
    fn default() -> Self {
        Self {
            frame_step: 1,
            max_frames: 2000,
            fps_override: None,
            setup: Default::default(),
            shape: Default::default(),
            pose_iterations: 5,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AmassFitResult {
    pub clip: PoseClip,
    pub shape_error: Vec<f64>,
    pub frame_errors: Vec<Vec<f64>>,
    /// Empty entries when the caller left refinement disabled; otherwise the
    /// initial loss plus each native Adam step for that frame.
    pub frame_refinement_losses: Vec<Vec<f64>>,
    /// Refinement trace for the retained shape stage when refinement is enabled.
    pub shape_refinement_losses: Vec<f64>,
    pub source_frame_indices: Vec<usize>,
}
/// Native two-stage shape-on-rest then warm-started pose fitting. Requires an
/// explicitly supplied SMPL-X source model AND correspondence map; neither is
/// downloaded. Pose correctives are evaluated by the source model.
///
/// Caller fitting settings (including optional `post_gd` refinement, local/facial
/// clamps, `max_delta` and excluded phenotypes) are carried into the pose stage.
/// The pose stage replaces the phenotype initialization with the fitted shape,
/// freezes phenotype optimization, drops multistart (it is a shape search) and
/// disables the phenotype shape-prior term, since that term is constant while
/// phenotypes are frozen.
pub fn fit_amass(
    sequence: &AmassSequence,
    source: &SmplModel,
    target: &Anny,
    mapping: &VertexMap,
    options: &AmassFitOptions,
) -> Result<AmassFitResult> {
    use crate::inverter::{AnnyInverter, FitOptions};
    source.validate()?;
    sequence.validate()?;
    ensure(
        options.frame_step > 0
            && options.max_frames > 0
            && options.max_frames <= 100_000
            && options.pose_iterations > 0
            && options.pose_iterations <= 1000,
        "invalid AMASS fitting limits",
    )?;
    let fps = options
        .fps_override
        .or(sequence.fps)
        .ok_or_else(|| invalid("AMASS has no frame rate; provide fps_override"))?;
    ensure(fps.is_finite() && fps > 0., "invalid AMASS fitting fps")?;
    ensure(
        sequence.frame_count().div_ceil(options.frame_step) <= options.max_frames,
        "AMASS clip exceeds max_frames; explicitly raise the limit or frame_step",
    )?;
    let source_frame_indices = (0..sequence.frame_count())
        .step_by(options.frame_step)
        .take(options.max_frames)
        .collect::<Vec<_>>();
    ensure(!source_frame_indices.is_empty(), "empty AMASS sequence")?;
    let fitter = AnnyInverter::new(target, options.setup.clone())?;
    let rest = source.forward(&sequence.parameters(0, source, true)?)?;
    let rest = mapping.map(rest.get("vertices")?, target.data.vertex_count())?;
    let shape = fitter.fit(&rest, &options.shape)?;
    let mut initialization = FitOptions {
        initial_phenotype_kwargs: shape.parameters.phenotype_kwargs.clone(),
        initial_pose_parameters: target
            .pose_parameters(&shape.output, PoseParameterization::LocalBone)?
            .nested_json(),
        optimize_phenotypes: false,
        max_n_iters: Some(options.pose_iterations),
        ..options.shape.clone()
    };
    // Pose-only stage: multistart searches the shape space, and the phenotype
    // prior is constant (and rejected) while phenotypes are frozen.
    initialization.multistart.clear();
    initialization.post_gd_prior_weight = 0.;
    let mut clip = PoseClip {
        name: "AMASS fitted to Anny".into(),
        parameters: Parameters {
            phenotype_kwargs: initialization.initial_phenotype_kwargs.clone(),
            pose_parameterization: Some(PoseParameterization::LocalBone),
            ..Default::default()
        },
        frames: Vec::new(),
    };
    let mut errors = Vec::new();
    let mut refinement = Vec::new();
    for &frame in &source_frame_indices {
        let output = source.forward(&sequence.parameters(frame, source, false)?)?;
        let vertices = mapping.map(output.get("vertices")?, target.data.vertex_count())?;
        let fit = fitter.fit(&vertices, &initialization)?;
        let pose = target
            .pose_parameters(&fit.output, PoseParameterization::LocalBone)?
            .nested_json();
        initialization.initial_pose_parameters = pose.clone();
        errors.push(fit.mean_vertex_error);
        refinement.push(fit.post_gd_losses.clone());
        clip.frames.push(PoseFrame {
            time: frame as f64 / fps,
            pose_parameters: pose,
        });
    }
    clip.validate(target)?;
    Ok(AmassFitResult {
        clip,
        shape_error: shape.mean_vertex_error,
        frame_errors: errors,
        frame_refinement_losses: refinement,
        shape_refinement_losses: shape.post_gd_losses.clone(),
        source_frame_indices,
    })
}
