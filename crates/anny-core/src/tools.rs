// Port of NAVER Anny anthropometry, keypoints, pose transfer and collision helpers.
// Copyright (C) 2025 NAVER Corp. SPDX-License-Identifier: Apache-2.0
use crate::{
    assets::AssetStore,
    ensure,
    math::*,
    mesh::{self, MeshBvh},
    model::Anny,
    tensor::Kind,
    Error, Result, Tensor,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Measurements {
    pub height: Vec<f64>,
    pub waist_circumference: Vec<f64>,
    pub volume: Vec<f64>,
    pub mass: Vec<f64>,
    pub bmi: Vec<f64>,
}
pub struct Anthropometry {
    faces: Vec<[usize; 3]>,
    waist: Vec<usize>,
    vertices: usize,
}
impl Anthropometry {
    pub fn new(model: &Anny) -> Result<Self> {
        let faces = model.data.get("faces")?;
        ensure(
            faces.shape.len() == 2 && faces.shape[1] == 3,
            "anthropometry requires triangular faces",
        )?;
        let base = model.data.get("base_mesh_vertex_indices")?;
        let waist = BASE_MESH_WAIST_VERTICES
            .iter()
            .map(|&i| {
                base.data
                    .iter()
                    .position(|&v| v == i as f64)
                    .ok_or_else(|| Error::Invalid(format!("model is missing waist vertex {i}")))
            })
            .collect::<Result<_>>()?;
        let ids = faces.checked_indices(model.data.vertex_count(), "measurement faces")?;
        Ok(Self {
            faces: ids.chunks_exact(3).map(|a| [a[0], a[1], a[2]]).collect(),
            waist,
            vertices: model.data.vertex_count(),
        })
    }
    pub fn measure(&self, rest_vertices: &Tensor) -> Result<Measurements> {
        ensure(
            rest_vertices.shape.len() == 3 && rest_vertices.shape[1..] == [self.vertices, 3],
            "measurements require [B,V,3]",
        )?;
        rest_vertices.validate()?;
        let mut result = Measurements {
            height: vec![],
            waist_circumference: vec![],
            volume: vec![],
            mass: vec![],
            bmi: vec![],
        };
        for row in rest_vertices.data.chunks_exact(self.vertices * 3) {
            let vertices: Vec<_> = row.chunks_exact(3).map(vec3).collect();
            let min = vertices.iter().map(|v| v[2]).fold(f64::INFINITY, f64::min);
            let max = vertices
                .iter()
                .map(|v| v[2])
                .fold(f64::NEG_INFINITY, f64::max);
            let height = max - min;
            let mut waist = 0.;
            for i in 0..self.waist.len() {
                waist += (vertices[self.waist[i]]
                    - vertices[self.waist[(i + 1) % self.waist.len()]])
                .norm();
            }
            let volume = self
                .faces
                .iter()
                .map(|f| vertices[f[0]].cross(&vertices[f[1]]).dot(&vertices[f[2]]) / 6.)
                .sum::<f64>()
                .abs();
            let mass = volume * 980.;
            result.height.push(height);
            result.waist_circumference.push(waist);
            result.volume.push(volume);
            result.mass.push(mass);
            result.bmi.push(mass / (height * height));
        }
        Ok(result)
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeypointsRegressor {
    pub labels: Vec<String>,
    pub weights: Tensor,
    pub indices: Option<Tensor>,
}
impl KeypointsRegressor {
    pub fn coco(store: &AssetStore, model: &Anny, labels: Option<Vec<String>>) -> Result<Self> {
        let a = store.converted("keypoints/coco.pth")?;
        let payload = a.payload()?;
        let labels = labels.unwrap_or_else(|| {
            payload
                .as_object()
                .map(|m| m.keys().cloned().collect())
                .unwrap_or_default()
        });
        let n = model.data.vertex_count();
        let base = model.data.get("base_mesh_vertex_indices")?;
        let mut weights = Tensor::zeros(vec![labels.len(), n]);
        for (i, name) in labels.iter().enumerate() {
            let source = a.payload_tensor(name)?;
            let indices = base.checked_indices(source.data.len(), "keypoint base indices")?;
            for (v, &index) in indices.iter().enumerate() {
                weights.data[i * n + v] = source.data[index];
            }
            ensure(
                (weights.data[i * n..(i + 1) * n].iter().sum::<f64>() - 1.).abs() < 1e-3,
                format!("keypoint {name} weights do not sum to one on this topology"),
            )?;
        }
        Ok(Self {
            labels,
            weights,
            indices: None,
        })
    }
    pub fn regress(&self, vertices: &Tensor) -> Result<Tensor> {
        ensure(
            vertices.shape.len() == 3 && vertices.shape[2] == 3,
            "keypoints need [B,V,3]",
        )?;
        vertices.validate()?;
        self.weights.validate()?;
        ensure(
            self.weights.shape.len() == 2 && self.weights.shape[0] == self.labels.len(),
            "keypoint weight shape mismatch",
        )?;
        let (b, n, k, s) = (
            vertices.shape[0],
            vertices.shape[1],
            self.labels.len(),
            self.weights.shape[1],
        );
        let ids = if let Some(i) = &self.indices {
            i.expect_shape(&self.weights.shape, "sparse keypoint indices")?;
            i.checked_indices(n, "sparse keypoint indices")?
        } else {
            ensure(s == n, "dense keypoint weights must cover all vertices")?;
            (0..k).flat_map(|_| 0..n).collect()
        };
        let mut out = Tensor::zeros(vec![b, k, 3]);
        for bi in 0..b {
            for ki in 0..k {
                for si in 0..s {
                    let w = self.weights.data[ki * s + si];
                    if w != 0. {
                        let v = ids[ki * s + si];
                        for axis in 0..3 {
                            out.data[(bi * k + ki) * 3 + axis] +=
                                w * vertices.data[(bi * n + v) * 3 + axis];
                        }
                    }
                }
            }
        }
        Ok(out)
    }
}
/// Transfer between compatible rest meshes, matching target bones by name.
pub fn transfer_pose_parameters(
    source: &Anny,
    target: &Anny,
    parameters: &crate::Parameters,
    mode: crate::PoseParameterization,
) -> Result<Tensor> {
    let source_output = source.forward(parameters)?;
    let mut target_parameters = parameters.clone();
    target_parameters.pose_parameters = serde_json::Value::Null;
    target_parameters.pose_parameterization = None;
    let mut target_output = target.forward(&target_parameters)?;
    let sr = source_output.get("rest_vertices")?;
    let tr = target_output.get("rest_vertices")?;
    ensure(
        sr.shape == tr.shape
            && sr
                .data
                .iter()
                .zip(&tr.data)
                .all(|(a, b)| (a - b).abs() < 1e-6),
        "pose transfer requires matching rest mesh geometry",
    )?;
    let ids: Vec<_> = target
        .data
        .metadata
        .bone_labels
        .iter()
        .map(|name| {
            source
                .data
                .metadata
                .bone_labels
                .iter()
                .position(|n| n == name)
                .ok_or_else(|| Error::Invalid(format!("source rig has no target bone {name}")))
        })
        .collect::<Result<_>>()?;
    let poses = source_output.get("bone_poses")?;
    let source_rest = source_output.get("rest_bone_poses")?;
    let target_rest = target_output.get("rest_bone_poses")?;
    let b = poses.shape[0];
    let j = target.data.bone_count();
    let sj = source.data.bone_count();
    let mut desired = crate::model::identity_poses(b, j);
    for bi in 0..b {
        for (i, &si) in ids.iter().enumerate() {
            let a = (bi * sj + si) * 16;
            let r = ((bi % source_rest.shape[0]) * sj + si) * 16;
            let t = ((bi % target_rest.shape[0]) * j + i) * 16;
            let p = mat4(&poses.data[a..a + 16])
                * inverse_rigid(&mat4(&source_rest.data[r..r + 16]))
                * mat4(&target_rest.data[t..t + 16]);
            write4(
                &p,
                &mut desired.data[(bi * j + i) * 16..(bi * j + i + 1) * 16],
            );
        }
    }
    target_output.arrays.insert("bone_poses".into(), desired);
    target.pose_parameters(&target_output, mode)
}
/// SAT test deliberately retains upstream's 1e-6 edge-axis threshold.
pub fn triangle_intersects_sat(a: [Vec3; 3], b: [Vec3; 3]) -> bool {
    let ea = [a[1] - a[0], a[2] - a[0], a[2] - a[1]];
    let eb = [b[1] - b[0], b[2] - b[0], b[2] - b[1]];
    let separated = |axis: Vec3| {
        let project = |v: &[Vec3; 3]| {
            let d = v.map(|p| axis.dot(&p));
            (
                d.iter().copied().fold(f64::INFINITY, f64::min),
                d.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            )
        };
        let (mina, maxa) = project(&a);
        let (minb, maxb) = project(&b);
        maxa < minb || maxb < mina
    };
    for n in [ea[0].cross(&ea[1]), eb[0].cross(&eb[1])] {
        if n.norm_squared() > 0. && separated(n.normalize()) {
            return false;
        }
    }
    for x in &ea {
        for y in &eb {
            let axis = x.cross(y);
            if axis.norm_squared() > 1e-6 && separated(axis.normalize()) {
                return false;
            }
        }
    }
    true
}
/// Returns one intersecting face index per face (or -1), using the same
/// skinning-group exclusion rule as Python's SelfInterpenetrationModule.
/// Selection among multiple intersections is deterministic by face id, not GPU order.
pub struct SelfInterpenetrationModule {
    faces: Tensor,
    /// Per-face bone-label ids, sorted and deduplicated.
    ///
    /// These used to be `BTreeSet<String>`, which cost ~82k `String` clones to build and a string
    /// comparison walk on every candidate face pair. Label ids are interned once instead; the
    /// disjointness test is now an integer merge walk.
    masks: Vec<Vec<u32>>,
    n: usize,
}
impl SelfInterpenetrationModule {
    pub fn new(
        model: &Anny,
        group_toes: bool,
        group_eyes: bool,
        group_tongue: bool,
    ) -> Result<Self> {
        let mut d = crate::ModelData::default();
        d.put("faces", model.data.get("faces")?.clone());
        d.put(
            "template_vertices",
            model.data.get("template_vertices")?.clone(),
        );
        mesh::triangulate(&mut d)?;
        let faces = d.arrays.remove("faces").unwrap();
        let labels: Vec<_> = model
            .data
            .metadata
            .bone_labels
            .iter()
            .map(|s| {
                if group_toes && s.contains("toe") {
                    if s.ends_with(".L") {
                        "left_toes".to_string()
                    } else {
                        "right_toes".to_string()
                    }
                } else if (group_eyes && s.contains("eye"))
                    || (group_tongue && s.contains("tongue"))
                {
                    "head".into()
                } else {
                    s.clone()
                }
            })
            .collect();
        // Intern the label vocabulary once per module: masks are compared against each other far more
        // often than they are built.
        let mut vocabulary: HashMap<&str, u32> = HashMap::new();
        let mut interned = Vec::with_capacity(labels.len());
        for s in &labels {
            let next = vocabulary.len() as u32;
            interned.push(*vocabulary.entry(s.as_str()).or_insert(next));
        }
        let weights = model.data.get("vertex_bone_weights")?;
        let ids = model.data.get("vertex_bone_indices")?;
        let k = weights.shape[1];
        let n = model.data.vertex_count();
        let mut vm: Vec<Vec<u32>> = vec![Vec::new(); n];
        for v in 0..n {
            let mask = &mut vm[v];
            for s in 0..k {
                if weights.data[v * k + s] > 0. {
                    mask.push(interned[ids.data[v * k + s] as usize]);
                }
            }
            mask.sort_unstable();
            mask.dedup();
        }
        let masks = faces
            .data
            .chunks_exact(3)
            .map(|f| {
                let mut mask: Vec<u32> = f
                    .iter()
                    .flat_map(|&v| vm[v as usize].iter().copied())
                    .collect();
                mask.sort_unstable();
                mask.dedup();
                mask
            })
            .collect();
        Ok(Self { faces, masks, n })
    }
    pub fn forward(&self, vertices: &Tensor) -> Result<Tensor> {
        ensure(
            vertices.shape.len() == 3 && vertices.shape[1..] == [self.n, 3],
            "collision input must be [B,V,3]",
        )?;
        let f = self.faces.shape[0];
        let mut out = Tensor {
            shape: vec![vertices.shape[0], f],
            data: vec![-1.; vertices.shape[0] * f],
            kind: Kind::Index,
        };
        // Broad phase: one BVH query per face, with the traversal buffers reused across all 27,420
        // queries and across batches.
        //
        // The candidate set is deliberately kept exactly as it was: the BVH returns faces from leaf
        // nodes whose AABB overlaps the query, so it is a superset of true AABB overlaps. An exact
        // AABB sweep was written and measured faster (12.6 ms vs 22.1 ms of search) but produced 728
        // partners instead of 940, because `triangle_intersects_sat` skips edge-cross axes with
        // `norm_squared() <= 1e-6` and so reports near-degenerate pairs as intersecting even when the
        // two AABBs are disjoint. Those pairs are reachable only through the leaf-union superset.
        // Upstream's BVH query has the same behaviour, so narrowing the candidate set would trade
        // parity for speed. Reusing the buffers is free of that trade.
        let mut stack: Vec<usize> = Vec::new();
        let mut candidates: Vec<usize> = Vec::new();
        for (bi, row) in vertices.data.chunks_exact(self.n * 3).enumerate() {
            let v = Tensor::new(vec![self.n, 3], row.to_vec())?;
            let bvh = MeshBvh::new(&v, &self.faces)?;
            for i in 0..f {
                let tri = self.face_triangle(row, i);
                let mut lo = tri[0];
                let mut hi = tri[0];
                for v in &tri[1..] {
                    for k in 0..3 {
                        lo[k] = lo[k].min(v[k]);
                        hi[k] = hi[k].max(v[k]);
                    }
                }
                bvh.overlapping_faces_into(lo, hi, &mut stack, &mut candidates);
                candidates.sort_unstable();
                for &j in candidates.iter() {
                    if i == j || label_masks_intersect(&self.masks[i], &self.masks[j]) {
                        continue;
                    }
                    let other = self.face_triangle(row, j);
                    if triangle_intersects_sat(tri, other) {
                        out.data[bi * f + i] = j as f64;
                        break;
                    }
                }
            }
        }
        Ok(out)
    }

    /// The three vertices of face `i` from a batch row.
    fn face_triangle(&self, row: &[f64], i: usize) -> [Vec3; 3] {
        std::array::from_fn(|s| {
            let v = self.faces.data[i * 3 + s] as usize * 3;
            vec3(&row[v..v + 3])
        })
    }
}
/// Whether two sorted, deduplicated label-id masks share at least one label.
///
/// This is the inner test of the collision search and runs once per candidate face pair, so it is a
/// merge walk over integers rather than a set comparison over strings. The predicate is stated
/// positively on purpose: the previous `BTreeSet<String>::is_disjoint` call was used negated, and a
/// mask test that returns the opposite of its name is exactly how a silent inversion happens.
fn label_masks_intersect(a: &[u32], b: &[u32]) -> bool {
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i] < b[j] {
            i += 1;
        } else if a[i] > b[j] {
            j += 1;
        } else {
            return true;
        }
    }
    false
}

/// Mesh cleanup helpers use graph connectivity, not a third-party rendering engine.
pub fn symmetric_vertex_indices(
    vertices: &Tensor,
    axis: usize,
    threshold: f64,
) -> Result<Vec<usize>> {
    ensure(
        vertices.shape.len() == 2 && vertices.shape[1] == 3 && axis < 3 && threshold > 0.,
        "invalid symmetry input",
    )?;
    let v: Vec<_> = vertices.data.chunks_exact(3).map(vec3).collect();
    let mut out = Vec::new();
    for p in &v {
        let mut q = *p;
        q[axis] = -q[axis];
        let (i, dist) = v
            .iter()
            .enumerate()
            .map(|(i, p)| (i, (p - q).norm()))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        ensure(dist < threshold, "symmetric counterpart outside threshold")?;
        out.push(i);
    }
    ensure(
        out.iter().copied().collect::<BTreeSet<_>>().len() == out.len(),
        "symmetry mapping is not one-to-one",
    )?;
    Ok(out)
}
pub fn boundary_edges(faces: &Tensor) -> Result<Vec<(usize, usize)>> {
    Ok(mesh::edges(faces)?
        .into_iter()
        .filter(|(_, count)| *count == 1)
        .map(|(e, _)| e)
        .collect())
}
pub const BASE_MESH_WAIST_VERTICES: &[usize] = &[
    4121, 10763, 10760, 10757, 10777, 10776, 10779, 10780, 10778, 10781, 10771, 10773, 10772,
    10775, 10774, 10814, 10834, 10816, 10817, 10818, 10819, 10820, 10821, 4181, 4180, 4179, 4178,
    4177, 4176, 4175, 4196, 4173, 4131, 4132, 4129, 4130, 4128, 4138, 4135, 4137, 4136, 4133, 4134,
    4108, 4113, 4118,
];
