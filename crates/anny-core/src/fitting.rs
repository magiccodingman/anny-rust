//! Explicit correspondence-based or nearest-surface fitting from ordinary meshes.
//! This local ICP workflow is not a global scan-registration algorithm. Source
//! and target must already share meters, Z-up, and roughly corresponding poses.
use crate::{
    ensure,
    inverter::{AnnyInverter, FitOptions, FitResult, InverterOptions},
    math::{point, vec3, Mat4, Vec3},
    mesh::MeshBvh,
    scene::SurfaceMesh,
    Anny, Parameters, PoseParameterization, Result, Tensor,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Correspondence {
    Index,
    ClosestSurface,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MeshFitOptions {
    pub outer_iterations: usize,
    pub inner_iterations: usize,
    pub tolerance: f64,
    pub max_distance: Option<f64>,
    /// Row-major target-to-Anny affine transform; default is no alignment.
    pub target_transform: [[f64; 4]; 4],
    /// Optional explicit landmark initialization, instead of target_transform.
    pub landmarks: Option<Landmarks>,
    pub initial: FitOptions,
    pub inverter: InverterOptions,
}
impl Default for MeshFitOptions {
    fn default() -> Self {
        Self {
            outer_iterations: 5,
            inner_iterations: 2,
            tolerance: 1e-6,
            max_distance: None,
            target_transform: [
                [1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.],
            ],
            landmarks: None,
            initial: FitOptions::default(),
            inverter: InverterOptions::default(),
        }
    }
}
pub struct MeshFitResult {
    pub fit: FitResult,
    /// Mean forward surface distance for ICP, paired vertex distance for index mode.
    pub distances: Vec<f64>,
    pub accepted_iterations: usize,
}
impl MeshFitResult {
    pub fn to_json(&self) -> serde_json::Value {
        json!({"fit":self.fit.to_json(),"distances":self.distances,"accepted_iterations":self.accepted_iterations})
    }
}
/// Recover seam-split source positions ONLY when the caller explicitly selected
/// index correspondence. Repeated IDs must agree spatially and cover all vertices.
/// An OBJ numbering map is not proof of correspondence to an arbitrary Anny model.
pub fn corresponding_vertices(mesh: &SurfaceMesh, count: usize) -> Result<Tensor> {
    mesh.validate()?;
    if mesh.source_vertex_indices.is_empty() {
        ensure(
            mesh.positions.len() == count,
            "index fitting requires identical vertex count and ordering",
        )?;
        return Tensor::new(
            vec![1, count, 3],
            mesh.positions.iter().flatten().copied().collect(),
        );
    }
    let mut out = vec![None; count];
    for (&id, p) in mesh.source_vertex_indices.iter().zip(&mesh.positions) {
        let id = id as usize;
        ensure(id < count, "source vertex map does not match this model")?;
        if let Some(old) = out[id] {
            let old: [f64; 3] = old;
            ensure(
                (vec3(&old) - vec3(p)).norm() < 1e-5,
                "split vertices with the same source ID disagree",
            )?;
        } else {
            out[id] = Some(*p);
        }
    }
    ensure(
        out.iter().all(Option::is_some),
        "source vertex map does not cover every model vertex",
    )?;
    Tensor::new(
        vec![1, count, 3],
        out.into_iter().flat_map(|x| x.unwrap()).collect(),
    )
}
pub fn fit_mesh(
    model: &Anny,
    mesh: &SurfaceMesh,
    mode: Correspondence,
    options: &MeshFitOptions,
) -> Result<MeshFitResult> {
    mesh.validate()?;
    ensure(
        options.outer_iterations > 0
            && options.outer_iterations <= 1000
            && options.inner_iterations > 0
            && options.inner_iterations <= 1000,
        "mesh fitting iterations must be in 1..=1000",
    )?;
    ensure(
        options.tolerance.is_finite() && options.tolerance >= 0.,
        "invalid mesh fitting tolerance",
    )?;
    if let Some(d) = options.max_distance {
        ensure(
            d.is_finite() && d > 0.,
            "invalid maximum correspondence distance",
        )?;
    }
    let target_transform = if let Some(landmarks) = &options.landmarks {
        ensure(
            options.target_transform == MeshFitOptions::default().target_transform,
            "choose landmarks or target_transform, not both",
        )?;
        landmarks.align()?.target_transform
    } else {
        options.target_transform
    };
    let transform = Mat4::from_row_slice(
        &target_transform
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>(),
    );
    crate::math::checked_rigid(&transform)?;
    ensure(
        transform.fixed_view::<3, 3>(0, 0).determinant().abs() > 1e-12,
        "singular target transform",
    )?;
    let mut target = mesh.clone();
    for p in &mut target.positions {
        *p = point(&transform, &vec3(p)).into();
    }
    let fitter = AnnyInverter::new(model, options.inverter.clone())?;
    if matches!(mode, Correspondence::Index) {
        let vertices = corresponding_vertices(&target, model.data.vertex_count())?;
        let fit = fitter.fit(&vertices, &options.initial)?;
        return Ok(MeshFitResult {
            distances: fit.mean_vertex_error.clone(),
            accepted_iterations: 1,
            fit,
        });
    }
    let bvh = MeshBvh::new(&target.vertex_tensor()?, &target.face_tensor())?;
    let mut setup = options.initial.clone();
    ensure(
        setup.multistart.is_empty(),
        "closest-surface fitting uses one explicit initialization, not multistart",
    )?;
    setup.max_n_iters = Some(options.inner_iterations);
    let mut initial = setup.initial_phenotype_kwargs.clone();
    if initial.is_null() {
        initial = json!({"age":0.8});
    }
    if let Some(values) = initial.as_object_mut() {
        values.entry("age").or_insert(json!(0.8));
    }
    let parameters = Parameters {
        phenotype_kwargs: initial,
        pose_parameters: setup.initial_pose_parameters.clone(),
        pose_parameterization: Some(PoseParameterization::LocalBone),
        ..Default::default()
    };
    let first = model.forward(&parameters)?;
    ensure(
        first.get("vertices")?.shape[0] == 1,
        "surface fitting expects a single character",
    )?;
    let mut current = first.get("vertices")?.clone();
    let project = |v: &Tensor| -> Result<(Tensor, f64)> {
        let mut coords = Vec::with_capacity(v.data.len());
        let mut total = 0.;
        for p in v.data.chunks_exact(3) {
            let (distance, face, weights) = bvh.closest(vec3(p));
            ensure(
                face != usize::MAX && distance.is_finite(),
                "target has no usable triangles",
            )?;
            if let Some(max) = options.max_distance {
                ensure(
                    distance <= max,
                    "correspondence exceeds max_distance; provide better initial alignment",
                )?;
            }
            let ids = bvh.triangle_indices(face);
            let q: Vec3 = ids
                .iter()
                .zip(weights)
                .map(|(&i, w)| w * vec3(&target.positions[i]))
                .sum();
            coords.extend_from_slice(q.as_slice());
            total += distance;
        }
        Ok((
            Tensor::new(v.shape.clone(), coords)?,
            total / v.shape[1] as f64,
        ))
    };
    let mut best: Option<FitResult> = None;
    let mut distances = vec![project(&current)?.1];
    let mut accepted = 0;
    for _ in 0..options.outer_iterations {
        let (corresponding, _) = project(&current)?;
        let next = fitter.fit(&corresponding, &setup)?;
        // The fitter may return only referenced vertices for -full topology.
        let vertices = next.output.get("vertices")?;
        let distance = project(vertices)?.1;
        let previous = *distances.last().unwrap();
        if distance > previous + options.tolerance {
            break;
        }
        setup.initial_phenotype_kwargs = next.parameters.phenotype_kwargs.clone();
        setup.initial_pose_parameters = model
            .pose_parameters(&next.output, PoseParameterization::LocalBone)?
            .nested_json();
        current = vertices.clone();
        distances.push(distance);
        accepted += 1;
        best = Some(next);
        if (previous - distance).abs() <= options.tolerance {
            break;
        }
    }
    let fit = best.unwrap_or(FitResult {
        parameters,
        vertices: current,
        output: first,
        mean_vertex_error: vec![distances[0]],
        iterations: 0,
    });
    Ok(MeshFitResult {
        fit,
        distances,
        accepted_iterations: accepted,
    })
}

/// Paired landmarks in original target-mesh coordinates and desired model space.
/// At least three non-collinear, positively weighted pairs are required. The
/// returned transform is a proper rigid (optionally uniform-scale) transform.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Landmarks {
    pub target_points: Vec<[f64; 3]>,
    pub model_points: Vec<[f64; 3]>,
    #[serde(default)]
    pub weights: Vec<f64>,
    #[serde(default)]
    pub estimate_scale: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Alignment {
    pub target_transform: [[f64; 4]; 4],
    pub scale: f64,
    pub weighted_rms: f64,
}
impl Landmarks {
    /// Weighted proper-rotation Procrustes alignment, with reflection correction.
    /// This initialization uses supplied correspondences; it does not discover
    /// landmarks or infer the correct orientation of an arbitrary scan.
    pub fn align(&self) -> Result<Alignment> {
        use crate::math::{rigid, Mat3};
        let n = self.target_points.len();
        ensure(
            (3..=100_000).contains(&n) && n == self.model_points.len(),
            "landmarks require 3..=100000 paired points",
        )?;
        ensure(
            self.weights.is_empty() || self.weights.len() == n,
            "landmark weight count mismatch",
        )?;
        ensure(
            self.target_points
                .iter()
                .flatten()
                .chain(self.model_points.iter().flatten())
                .all(|x| x.is_finite()),
            "non-finite landmark",
        )?;
        let mut weights = if self.weights.is_empty() {
            vec![1.; n]
        } else {
            self.weights.clone()
        };
        ensure(
            weights.iter().all(|w| w.is_finite() && *w >= 0.)
                && weights.iter().filter(|w| **w > 0.).count() >= 3,
            "need three positive finite landmark weights",
        )?;
        let total: f64 = weights.iter().sum();
        ensure(
            total.is_finite() && total > 0.,
            "invalid landmark weight sum",
        )?;
        for w in &mut weights {
            *w /= total;
        }
        let mut ca = Vec3::zeros();
        let mut cb = Vec3::zeros();
        for ((a, b), w) in self
            .target_points
            .iter()
            .zip(&self.model_points)
            .zip(&weights)
        {
            ca += vec3(a) * *w;
            cb += vec3(b) * *w;
        }
        let mut covariance = Mat3::zeros();
        let mut variance = 0.;
        for ((a, b), w) in self
            .target_points
            .iter()
            .zip(&self.model_points)
            .zip(&weights)
        {
            let a = vec3(a) - ca;
            let b = vec3(b) - cb;
            covariance += b * a.transpose() * *w;
            variance += a.norm_squared() * *w;
        }
        ensure(
            covariance.iter().all(|x| x.is_finite()) && variance.is_finite() && variance > 0.,
            "degenerate or overflowing landmark covariance",
        )?;
        let svd = covariance.svd(true, true);
        ensure(
            svd.singular_values[0] > 0. && svd.singular_values[1] > svd.singular_values[0] * 1e-10,
            "landmarks are collinear or rank deficient",
        )?;
        let u = svd.u.unwrap();
        let vt = svd.v_t.unwrap();
        let sign = if (u * vt).determinant() < 0. { -1. } else { 1. };
        let mut correction = Mat3::identity();
        correction[(2, 2)] = sign;
        let rotation = u * correction * vt;
        let scale = if self.estimate_scale {
            (svd.singular_values[0] + svd.singular_values[1] + sign * svd.singular_values[2])
                / variance
        } else {
            1.
        };
        ensure(scale.is_finite() && scale > 0., "invalid landmark scale")?;
        let transform = rigid(&(rotation * scale), &(cb - rotation * ca * scale));
        ensure(
            transform.iter().all(|x| x.is_finite()),
            "overflowing landmark transform",
        )?;
        let rms = self
            .target_points
            .iter()
            .zip(&self.model_points)
            .zip(&weights)
            .map(|((a, b), w)| (point(&transform, &vec3(a)) - vec3(b)).norm_squared() * *w)
            .sum::<f64>()
            .sqrt();
        ensure(rms.is_finite(), "invalid landmark residual")?;
        Ok(Alignment {
            target_transform: std::array::from_fn(|i| std::array::from_fn(|j| transform[(i, j)])),
            scale,
            weighted_rms: rms,
        })
    }
}
