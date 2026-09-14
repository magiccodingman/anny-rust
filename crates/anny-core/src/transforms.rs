//! Reusable, transactional ModelData authoring operations.
//! Port of NAVER Anny model_transforms.py, Copyright (C) 2025 NAVER Corp.
//! SPDX-License-Identifier: Apache-2.0
use crate::{
    assets, config::BoneOrientation, ensure, mesh, model::ModelData, Error, Result, Tensor,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

fn transaction(
    data: &ModelData,
    f: impl FnOnce(&mut ModelData) -> Result<()>,
) -> Result<ModelData> {
    data.validate()?;
    let mut out = data.clone();
    f(&mut out)?;
    out.validate()?;
    Ok(out)
}
/// Select exact blendshape rows while keeping every corresponding bone buffer.
/// Local changes are signed pairs; selecting only half a pair is rejected.
pub fn filter_blendshapes(data: &ModelData, mask: &[bool]) -> Result<ModelData> {
    ensure(
        mask.len() == data.blendshape_count(),
        "blendshape mask length mismatch",
    )?;
    let local: Vec<_> = data
        .metadata
        .blendshape_labels
        .iter()
        .enumerate()
        .filter(|(_, n)| n.starts_with("local_change:"))
        .map(|(i, _)| i)
        .collect();
    ensure(
        local.len().is_multiple_of(2),
        "unpaired source local changes",
    )?;
    for pair in local.chunks_exact(2) {
        ensure(
            mask[pair[0]] == mask[pair[1]],
            "local changes must retain or remove both signed rows",
        )?;
    }
    transaction(data, |d| {
        let selected: Vec<_> = mask
            .iter()
            .enumerate()
            .filter(|(_, x)| **x)
            .map(|(i, _)| i)
            .collect();
        for key in [
            "blendshapes",
            "bone_heads_blendshapes",
            "bone_tails_blendshapes",
            "bone_orientation_blendshapes",
        ] {
            if let Some(t) = d.arrays.get(key) {
                d.put(key, t.select(0, &selected)?);
            }
        }
        if let Some(t) = d.arrays.get("stacked_phenotype_blend_shapes_mask") {
            ensure(
                t.shape.len() == 2 && t.shape[0] <= mask.len(),
                "invalid phenotype mask shape",
            )?;
            let macro_rows: Vec<_> = selected
                .iter()
                .copied()
                .filter(|&i| i < t.shape[0])
                .collect();
            d.put(
                "stacked_phenotype_blend_shapes_mask",
                t.select(0, &macro_rows)?,
            );
        }
        d.metadata.blendshape_labels = selected
            .iter()
            .map(|&i| data.metadata.blendshape_labels[i].clone())
            .collect();
        Ok(())
    })
}
pub fn filter_faces(data: &ModelData, keep: &[usize]) -> Result<ModelData> {
    transaction(data, |d| mesh::filter_faces(d, keep))
}
pub fn triangulate(data: &ModelData) -> Result<ModelData> {
    transaction(data, mesh::triangulate)
}
/// Only the un-reindexed original MakeHuman quad convention accepts authored caps.
pub fn edit_mesh(data: &ModelData) -> Result<ModelData> {
    ensure(
        data.get("base_mesh_vertex_indices")?
            .data
            .iter()
            .enumerate()
            .all(|(i, &v)| v == i as f64),
        "authored mesh edits require the original vertex ordering",
    )?;
    transaction(data, mesh::edit_mesh)
}
/// Compact mesh vertices and compose the existing source-vertex map.
/// A runtime Procrustes sample with nonzero weight may not be removed; prepare
/// cached orientations first when removing geometry that determines a bone frame.
pub fn remove_unattached_vertices(data: &ModelData) -> Result<ModelData> {
    transaction(data, |d| {
        let n = d.vertex_count();
        let keep: Vec<_> = d
            .get("faces")?
            .checked_indices(n, "faces")?
            .into_iter()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let mut map = vec![None; n];
        for (i, &old) in keep.iter().enumerate() {
            map[old] = Some(i);
        }
        let updated = if let Some(samples) = d.arrays.get("bone_vertex_indices") {
            let weights = d.get("bone_vertex_weights")?;
            weights.expect_shape(&samples.shape, "Procrustes sample weights")?;
            let mut s = samples.clone();
            for (i, old) in samples
                .checked_indices(n, "Procrustes samples")?
                .into_iter()
                .enumerate()
            {
                ensure(map[old].is_some() || weights.data[i] == 0., "removed vertex influences runtime orientation; cache orientations before pruning")?;
                s.data[i] = map[old].unwrap_or(0) as f64;
            }
            Some(s)
        } else {
            None
        };
        mesh::remove_unattached_vertices(d)?;
        if let Some(s) = updated {
            d.put("bone_vertex_indices", s);
        }
        Ok(())
    })
}
pub fn compact_skinning_weights(data: &ModelData) -> Result<ModelData> {
    transaction(data, mesh::compact_skinning_weights)
}
/// Bone-name mirror conventions used by MakeHuman, Mixamo and SOMA.
pub fn symmetric_bone_name(name: &str) -> String {
    if let Some(base) = name.strip_suffix(".L") {
        return format!("{base}.R");
    }
    if let Some(base) = name.strip_suffix(".R") {
        return format!("{base}.L");
    }
    if name.contains("Left") {
        return name.replacen("Left", "Right", 1);
    }
    if name.contains("Right") {
        return name.replacen("Right", "Left", 1);
    }
    name.to_owned()
}
/// Nearest reflected vertex with deterministic tie-breaking and bijection checks.
/// Spatial cells avoid the original all-pairs O(N²) distance matrix.
pub fn symmetric_vertex_indices(
    vertices: &Tensor,
    axis: usize,
    tolerance: f64,
) -> Result<Vec<usize>> {
    ensure(
        vertices.shape.len() == 2 && vertices.shape[1] == 3 && axis < 3,
        "symmetry needs N x 3 vertices and an axis in 0..3",
    )?;
    vertices.validate()?;
    ensure(
        tolerance.is_finite() && tolerance > 0.,
        "symmetry tolerance must be positive",
    )?;
    let key = |v: &[f64]| -> Result<[i64; 3]> {
        let mut k = [0; 3];
        for a in 0..3 {
            let f = (v[a] / tolerance).floor();
            ensure(
                f.abs() < (i64::MAX / 2) as f64,
                "symmetry coordinate overflow",
            )?;
            k[a] = f as i64;
        }
        Ok(k)
    };
    let mut grid: BTreeMap<[i64; 3], Vec<usize>> = BTreeMap::new();
    for (i, v) in vertices.data.chunks_exact(3).enumerate() {
        grid.entry(key(v)?).or_default().push(i);
    }
    let mut out = Vec::with_capacity(vertices.shape[0]);
    for v in vertices.data.chunks_exact(3) {
        let mut reflected = [v[0], v[1], v[2]];
        reflected[axis] = -reflected[axis];
        let cell = key(&reflected)?;
        let mut best = (f64::INFINITY, usize::MAX);
        for x in -1..=1 {
            for y in -1..=1 {
                for z in -1..=1 {
                    if let Some(candidates) = grid.get(&[cell[0] + x, cell[1] + y, cell[2] + z]) {
                        for &i in candidates {
                            let q = &vertices.data[i * 3..i * 3 + 3];
                            let distance: f64 = (0..3).map(|a| (q[a] - reflected[a]).powi(2)).sum();
                            if distance < best.0 || (distance == best.0 && i < best.1) {
                                best = (distance, i);
                            }
                        }
                    }
                }
            }
        }
        ensure(
            best.0 <= tolerance * tolerance,
            "mesh has no mirrored vertex within tolerance",
        )?;
        out.push(best.1);
    }
    ensure(
        out.iter().collect::<BTreeSet<_>>().len() == out.len()
            && out.iter().enumerate().all(|(i, &j)| out[j] == i),
        "mesh reflection is not a one-to-one involution",
    )?;
    Ok(out)
}
/// Average sparse weights over reflected vertices and corresponding mirrored bones.
/// Positive weights are sorted descending with stable bone-index tie-breaking.
pub fn symmetrize_skinning_weights(data: &ModelData) -> Result<ModelData> {
    data.validate()?;
    let sym = symmetric_vertex_indices(data.get("template_vertices")?, 0, 1e-4)?;
    let names: BTreeMap<_, _> = data
        .metadata
        .bone_labels
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();
    let mirror: Vec<_> = data
        .metadata
        .bone_labels
        .iter()
        .map(|n| {
            names
                .get(symmetric_bone_name(n).as_str())
                .copied()
                .ok_or_else(|| Error::Invalid(format!("bone {n} has no mirror counterpart")))
        })
        .collect::<Result<_>>()?;
    let n = data.vertex_count();
    let j = data.bone_count();
    let w = data.get("vertex_bone_weights")?;
    let ids = data
        .get("vertex_bone_indices")?
        .checked_indices(j, "skinning")?;
    let k = w.shape[1];
    let mut rows = Vec::with_capacity(n);
    let mut width = 1;
    for (v, &mate) in sym.iter().enumerate() {
        let mut row = vec![0.; j];
        for s in 0..k {
            row[ids[v * k + s]] += 0.5 * w.data[v * k + s];
            row[mirror[ids[mate * k + s]]] += 0.5 * w.data[mate * k + s];
        }
        let mut row: Vec<_> = row
            .into_iter()
            .enumerate()
            .filter(|(_, w)| *w > 0.)
            .collect();
        row.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        width = width.max(row.len());
        rows.push(row);
    }
    transaction(data, |d| {
        let mut out_w = Tensor::zeros(vec![n, width]);
        let mut out_i = Tensor::indices(vec![n, width], vec![0; n * width]);
        for (v, row) in rows.iter().enumerate() {
            for (s, &(b, w)) in row.iter().enumerate() {
                out_i.data[v * width + s] = b as f64;
                out_w.data[v * width + s] = w;
            }
        }
        d.put("vertex_bone_weights", out_w);
        d.put("vertex_bone_indices", out_i);
        Ok(())
    })
}
fn components(adjacency: &[Vec<usize>], allowed: &[bool]) -> Vec<Vec<usize>> {
    let mut visited = vec![false; allowed.len()];
    let mut output = Vec::new();
    for start in 0..allowed.len() {
        if !allowed[start] || visited[start] {
            continue;
        }
        let mut queue = VecDeque::from([start]);
        visited[start] = true;
        let mut current = vec![];
        while let Some(i) = queue.pop_front() {
            current.push(i);
            for &next in &adjacency[i] {
                if allowed[next] && !visited[next] {
                    visited[next] = true;
                    queue.push_back(next);
                }
            }
        }
        current.sort_unstable();
        output.push(current);
    }
    output.sort_by(|a, b| b.len().cmp(&a.len()).then(a[0].cmp(&b[0])));
    output
}
/// Remove disconnected per-bone influence islands in the largest mesh component.
/// Rows left completely unbound fall back to their original weights, like upstream.
pub fn remove_skinning_islands(data: &ModelData) -> Result<ModelData> {
    data.validate()?;
    let n = data.vertex_count();
    let j = data.bone_count();
    let mut adjacency = vec![Vec::new(); n];
    let mut used = vec![false; n];
    for &(a, b) in mesh::edges(data.get("faces")?)?.keys() {
        adjacency[a].push(b);
        adjacency[b].push(a);
        used[a] = true;
        used[b] = true;
    }
    let mesh_components = components(&adjacency, &used);
    ensure(
        !mesh_components.is_empty(),
        "island cleanup needs at least one face",
    )?;
    let mut body = vec![false; n];
    for &v in &mesh_components[0] {
        body[v] = true;
    }
    let ids = data
        .get("vertex_bone_indices")?
        .checked_indices(j, "weights")?;
    let original = data.get("vertex_bone_weights")?;
    let k = original.shape[1];
    transaction(data, |d| {
        let mut weights = original.clone();
        for bone in 0..j {
            let mask: Vec<_> = (0..n)
                .map(|v| {
                    body[v]
                        && (0..k).any(|s| ids[v * k + s] == bone && original.data[v * k + s] > 0.)
                })
                .collect();
            let parts = components(&adjacency, &mask);
            if parts.len() < 2 {
                continue;
            }
            for part in parts.iter().skip(1) {
                for &v in part {
                    for s in 0..k {
                        if ids[v * k + s] == bone {
                            weights.data[v * k + s] = 0.;
                        }
                    }
                }
            }
        }
        for (row, old) in weights
            .data
            .chunks_exact_mut(k)
            .zip(original.data.chunks_exact(k))
        {
            let mut sum: f64 = row.iter().sum();
            if sum <= 0. {
                row.copy_from_slice(old);
                sum = row.iter().sum();
            }
            ensure(sum > 0., "island cleanup left an unbound vertex")?;
            for w in row {
                *w /= sum;
            }
        }
        d.put("vertex_bone_weights", weights);
        Ok(())
    })
}
/// Filter/reparent bones, carrying cached covariance rows as well as bind data.
/// SOMA runtime child-offset refinement requires rebuilding its authored rig;
/// upstream likewise does not support arbitrary SOMA rig modifiers.
pub fn filter_rig(
    data: &ModelData,
    remove: &BTreeSet<String>,
    subtree: Option<&str>,
) -> Result<ModelData> {
    ensure(
        !data.arrays.contains_key("bone_children_indices"),
        "rebuild the SOMA rig instead of pruning child-offset refinement data",
    )?;
    ensure(
        !data.arrays.contains_key("bone_vertex_indices"),
        "cache runtime Procrustes orientations before changing the rig",
    )?;
    for name in remove {
        ensure(
            data.metadata.bone_labels.contains(name),
            format!("unknown bone {name}"),
        )?;
    }
    transaction(data, |d| {
        if let Some(t) = d.arrays.get_mut("bone_rolls_rotmat") {
            if t.shape.len() == 3 {
                t.shape.insert(0, 1);
            }
        }
        assets::filter_rig(d, remove, subtree)?;
        let selection: Vec<_> = d
            .metadata
            .bone_labels
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let source_name = if i == 0 {
                    subtree.unwrap_or(name)
                } else {
                    name
                };
                data.metadata
                    .bone_labels
                    .iter()
                    .position(|n| n == source_name)
                    .ok_or_else(|| Error::Invalid("bone remap failed".into()))
            })
            .collect::<Result<_>>()?;
        for (key, axis) in [
            ("bone_template_orientation_matrices", 0),
            ("bone_orientation_blendshapes", 1),
            ("reference_bone_orientations", 0),
        ] {
            if let Some(t) = data.arrays.get(key) {
                d.put(key, t.select(axis, &selection)?);
            }
        }
        Ok(())
    })
}
pub fn apply_procrustes_orientation(data: &ModelData) -> Result<ModelData> {
    transaction(data, |d| {
        d.remove(&[
            "bone_template_orientation_matrices",
            "bone_orientation_blendshapes",
            "bone_children_indices",
            "bone_children_mask",
            "bone_children_local_offsets",
        ]);
        assets::procrustes_orientation(d)
    })
}
/// Barycentric resampling, with optional exact target template and UVs.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Retopology {
    pub indices: Tensor,
    pub weights: Tensor,
    pub faces: Tensor,
    #[serde(default)]
    pub template_vertices: Option<Tensor>,
    #[serde(default)]
    pub base_mesh_vertex_indices: Option<Tensor>,
    #[serde(default)]
    pub texture_coordinates: Option<Tensor>,
    #[serde(default)]
    pub face_texture_coordinate_indices: Option<Tensor>,
}
pub fn apply_retopology(data: &ModelData, mapping: &Retopology) -> Result<ModelData> {
    transaction(data, |d| {
        assets::interpolate_model_data(
            d,
            &mapping.indices,
            &mapping.weights,
            mapping.faces.clone(),
            mapping.template_vertices.clone(),
            mapping.base_mesh_vertex_indices.clone(),
            true,
        )?;
        ensure(
            mapping.texture_coordinates.is_some()
                == mapping.face_texture_coordinate_indices.is_some(),
            "retopology needs both UV coordinates and face UV indices",
        )?;
        if let Some(t) = &mapping.texture_coordinates {
            d.put("texture_coordinates", t.clone());
        }
        if let Some(t) = &mapping.face_texture_coordinate_indices {
            d.put("face_texture_coordinate_indices", t.clone());
        }
        Ok(())
    })
}
/// Project a supplied target mesh onto a known corresponding reference shape.
/// The template remains a resampling of Anny; target vertex offsets are not
/// silently substituted for a different phenotype's template.
pub fn retopology_from_mesh(
    data: &ModelData,
    target: &Tensor,
    faces: Tensor,
    reference: &Tensor,
    max_distance: Option<f64>,
) -> Result<ModelData> {
    reference.expect_shape(&[data.vertex_count(), 3], "retopology reference")?;
    ensure(
        target.shape.len() == 2 && target.shape[1] == 3,
        "retopology target must be N x 3",
    )?;
    let (indices, weights) =
        assets::projection_coordinates(target, reference, data.get("faces")?, max_distance)?;
    apply_retopology(
        data,
        &Retopology {
            indices,
            weights,
            faces,
            template_vertices: None,
            base_mesh_vertex_indices: None,
            texture_coordinates: None,
            face_texture_coordinate_indices: None,
        },
    )
}
/// Explicit authoring pipeline used by the CLI and all language bindings.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "kebab-case", deny_unknown_fields)]
pub enum Transform {
    Triangulate,
    EditMesh,
    FilterFaces {
        indices: Vec<usize>,
    },
    FilterBlendshapes {
        labels: Vec<String>,
    },
    RemoveUnattachedVertices,
    SymmetrizeSkinningWeights,
    RemoveSkinningIslands,
    CompactSkinningWeights,
    FilterRig {
        #[serde(default)]
        remove: BTreeSet<String>,
        #[serde(default)]
        subtree: Option<String>,
    },
    ProcrustesOrientation,
    Retopology {
        mapping: Box<Retopology>,
    },
}
impl Transform {
    pub fn apply(&self, d: &ModelData) -> Result<ModelData> {
        match self {
            Self::Triangulate => triangulate(d),
            Self::EditMesh => edit_mesh(d),
            Self::FilterFaces { indices } => filter_faces(d, indices),
            Self::FilterBlendshapes { labels } => {
                ensure(
                    labels.iter().collect::<BTreeSet<_>>().len() == labels.len(),
                    "duplicate blendshape label",
                )?;
                for name in labels {
                    ensure(
                        d.metadata.blendshape_labels.contains(name),
                        format!("unknown blendshape {name}"),
                    )?;
                }
                filter_blendshapes(
                    d,
                    &d.metadata
                        .blendshape_labels
                        .iter()
                        .map(|s| labels.contains(s))
                        .collect::<Vec<_>>(),
                )
            }
            Self::RemoveUnattachedVertices => remove_unattached_vertices(d),
            Self::SymmetrizeSkinningWeights => symmetrize_skinning_weights(d),
            Self::RemoveSkinningIslands => remove_skinning_islands(d),
            Self::CompactSkinningWeights => compact_skinning_weights(d),
            Self::FilterRig { remove, subtree } => filter_rig(d, remove, subtree.as_deref()),
            Self::ProcrustesOrientation => apply_procrustes_orientation(d),
            Self::Retopology { mapping } => apply_retopology(d, mapping),
        }
    }
}
/// Preserve the loaded model's scalar behavior/configuration while authoring data.
pub fn apply_pipeline(model: &crate::Anny, operations: &[Transform]) -> Result<crate::Anny> {
    let mut data = model.data.clone();
    let mut config = model.config.clone();
    for operation in operations {
        data = operation.apply(&data)?;
        if matches!(operation, Transform::ProcrustesOrientation) {
            let mut rig = config.rig.resolve()?;
            rig.bone_orientation = BoneOrientation::Procrustes;
            config.rig = crate::config::RigSpec::Config(rig);
        }
    }
    crate::Anny::from_model_data(data, config)
}
/// MakeHuman-compatible sparse weight JSON, preserving original metadata and raw
/// vertex IDs even after mesh compaction. No source files are modified.
pub fn export_weights(
    data: &ModelData,
    source_header: &serde_json::Value,
) -> Result<serde_json::Value> {
    data.validate()?;
    let mut out = source_header
        .as_object()
        .cloned()
        .ok_or_else(|| Error::Invalid("weights header must be an object".into()))?;
    let mut weights = serde_json::Map::new();
    if let Some(old) = out.get("weights").and_then(|v| v.as_object()) {
        for name in old.keys() {
            weights.insert(name.clone(), serde_json::json!([]));
        }
    }
    let ids = data
        .get("vertex_bone_indices")?
        .checked_indices(data.bone_count(), "bone weights")?;
    let w = data.get("vertex_bone_weights")?;
    let k = w.shape[1];
    let base = data.get("base_mesh_vertex_indices")?;
    ensure(
        base.data
            .iter()
            .all(|&x| x >= 0. && x.fract() == 0. && x <= (1u64 << 53) as f64),
        "invalid raw source vertex indices",
    )?;
    for v in 0..data.vertex_count() {
        for s in 0..k {
            let value = w.data[v * k + s];
            if value > 0. {
                let name = &data.metadata.bone_labels[ids[v * k + s]];
                weights
                    .entry(name.clone())
                    .or_insert_with(|| serde_json::json!([]))
                    .as_array_mut()
                    .unwrap()
                    .push(serde_json::json!([base.data[v] as u64, value]));
            }
        }
    }
    out.insert("name".into(), serde_json::json!("Anny skinning weights"));
    out.insert(
        "description".into(),
        serde_json::json!("Native Rust authoring export; original license metadata retained."),
    );
    out.insert("weights".into(), serde_json::Value::Object(weights));
    Ok(serde_json::Value::Object(out))
}

/// Packed source-weight interpolation, including signed non-barycentric
/// coordinates. Packing follows the first appearance of each source bone.
#[derive(Clone, Debug)]
pub struct InterpolatedSkinning {
    pub indices: Tensor,
    pub weights: Tensor,
}
pub fn interpolate_skinning_weights(
    source: &ModelData,
    references: &Tensor,
    coefficients: &Tensor,
    check_weights: bool,
) -> Result<InterpolatedSkinning> {
    source.validate()?;
    ensure(
        references.shape.len() == 2 && references.shape[0] > 0 && references.shape[1] > 0,
        "reference indices must be N x K",
    )?;
    coefficients.expect_shape(&references.shape, "weight interpolation coefficients")?;
    let ids = references.checked_indices(source.vertex_count(), "weight interpolation")?;
    let old = source
        .get("vertex_bone_indices")?
        .checked_indices(source.bone_count(), "source skinning")?;
    let old_w = source.get("vertex_bone_weights")?;
    let slots = old_w.shape[1];
    let n = references.shape[0];
    let k = references.shape[1];
    let mut rows = Vec::with_capacity(n);
    let mut width = 1;
    for v in 0..n {
        let mut totals = vec![0.; source.bone_count()];
        let mut seen = vec![false; source.bone_count()];
        let mut order = vec![];
        for r in 0..k {
            let src = ids[v * k + r];
            for s in 0..slots {
                let b = old[src * slots + s];
                if !seen[b] {
                    seen[b] = true;
                    order.push(b);
                }
                totals[b] += coefficients.data[v * k + r] * old_w.data[src * slots + s];
            }
        }
        let mut row: Vec<_> = order
            .into_iter()
            .filter(|&b| totals[b].abs() > 1e-12)
            .map(|b| (b, totals[b]))
            .collect();
        let sum = row.iter().map(|x| x.1).sum::<f64>();
        if sum.abs() < 1e-12 {
            ensure(
                !check_weights,
                format!("interpolated target {v} is unbound"),
            )?;
            for (_, w) in &mut row {
                *w = 0.;
            }
        } else {
            for (_, w) in &mut row {
                *w /= sum;
            }
        }
        width = width.max(row.len());
        rows.push(row);
    }
    let mut indices = Tensor::indices(vec![n, width], vec![0; n * width]);
    let mut weights = Tensor::zeros(vec![n, width]);
    for (v, row) in rows.iter().enumerate() {
        for (s, &(b, w)) in row.iter().enumerate() {
            indices.data[v * width + s] = b as f64;
            weights.data[v * width + s] = w;
        }
    }
    Ok(InterpolatedSkinning { indices, weights })
}
/// Convex resampling also supports a point set (faces=None), as upstream does.
pub fn interpolate_model_data(
    data: &ModelData,
    indices: &Tensor,
    weights: &Tensor,
    faces: Option<Tensor>,
    base_indices: Option<Tensor>,
) -> Result<ModelData> {
    apply_retopology(
        data,
        &Retopology {
            indices: indices.clone(),
            weights: weights.clone(),
            faces: faces.unwrap_or_else(|| Tensor::indices(vec![0, 3], vec![])),
            template_vertices: None,
            base_mesh_vertex_indices: base_indices,
            texture_coordinates: None,
            face_texture_coordinate_indices: None,
        },
    )
}
/// Reproject legacy runtime-Procrustes samples onto a target surface. Cached
/// covariance models simply carry their topology-independent orientation data.
pub fn apply_procrustes_retopology(
    data: &ModelData,
    vertices: &Tensor,
    faces: &Tensor,
    indices: &Tensor,
    weights: &Tensor,
    base_indices: Option<Tensor>,
) -> Result<ModelData> {
    use crate::math::*;
    data.validate()?;
    ensure(
        vertices.shape.len() == 2 && vertices.shape[1] == 3,
        "target vertices must be N x 3",
    )?;
    let cached = data
        .arrays
        .contains_key("bone_template_orientation_matrices");
    let mut target = data.clone();
    target.remove(&[
        "bone_nonzeroweight_mask",
        "bone_vertex_indices",
        "bone_vertex_weights",
        "template_bone_vertices",
    ]);
    assets::interpolate_model_data(
        &mut target,
        indices,
        weights,
        faces.clone(),
        Some(vertices.clone()),
        base_indices,
        false,
    )?;
    if cached {
        target.validate()?;
        return Ok(target);
    }
    let active = data.get("bone_nonzeroweight_mask")?;
    active.expect_shape(&[data.bone_count()], "Procrustes active bones")?;
    let ids = data.get("bone_vertex_indices")?;
    let sample_weights = data.get("bone_vertex_weights")?;
    sample_weights.expect_shape(&ids.shape, "Procrustes sample weights")?;
    let ids_flat = ids.checked_indices(data.vertex_count(), "Procrustes samples")?;
    let bones: Vec<_> = active
        .data
        .iter()
        .enumerate()
        .filter_map(|(b, &x)| (x != 0.).then_some(b))
        .collect();
    ensure(
        ids.shape.len() == 2 && ids.shape[0] == bones.len(),
        "Procrustes sample row count mismatch",
    )?;
    ensure(
        sample_weights.data.iter().all(|&x| x >= 0.),
        "Procrustes sample weights must be nonnegative",
    )?;
    let (ref_ids, ref_weights) = assets::projection_coordinates(
        data.get("template_vertices")?,
        vertices,
        faces,
        Some(1000.),
    )?;
    let projected = ref_ids.checked_indices(vertices.shape[0], "projected sample")?;
    let source_rest = crate::model::rest_model(
        data,
        &crate::config::RigConfig {
            bone_orientation: BoneOrientation::Procrustes,
            ..crate::config::RigConfig::parse("makehuman")?
        },
        &Tensor::zeros(vec![1, data.blendshape_count()]),
    )?;
    let poses = source_rest.get("rest_bone_poses")?;
    let mut rows = vec![];
    let mut max = 0;
    for row in 0..bones.len() {
        let mut positions = BTreeMap::<usize, usize>::new();
        let mut accum: Vec<(usize, f64)> = vec![];
        for s in 0..ids.shape[1] {
            let source = ids_flat[row * ids.shape[1] + s];
            let weight = sample_weights.data[row * ids.shape[1] + s].sqrt();
            for k in 0..3 {
                let vertex = projected[source * 3 + k];
                let u = ref_weights.data[source * 3 + k];
                let slot = *positions.entry(vertex).or_insert_with(|| {
                    let n = accum.len();
                    accum.push((vertex, 0.));
                    n
                });
                accum[slot].1 += u * weight;
            }
        }
        max = max.max(accum.len());
        rows.push(accum);
    }
    let mut packed = Tensor::indices(vec![bones.len(), max], vec![0; bones.len() * max]);
    let mut ws = Tensor::zeros(vec![bones.len(), max]);
    let mut template = Tensor::zeros(vec![bones.len(), max, 3]);
    for (row, (&bone, values)) in bones.iter().zip(&rows).enumerate() {
        let inverse = inverse_rigid(&mat4(&poses.data[bone * 16..bone * 16 + 16]));
        for s in 0..max {
            let (v, w) = values.get(s).copied().unwrap_or((0, 0.));
            packed.data[row * max + s] = v as f64;
            ws.data[row * max + s] = w * w;
            let local = point(&inverse, &vec3(&vertices.data[v * 3..v * 3 + 3]));
            template.data[(row * max + s) * 3..(row * max + s + 1) * 3]
                .copy_from_slice(local.as_slice());
        }
    }
    target.put("bone_nonzeroweight_mask", active.clone());
    target.put("bone_vertex_indices", packed);
    target.put("bone_vertex_weights", ws);
    target.put("template_bone_vertices", template);
    target.validate()?;
    Ok(target)
}
/// Apply the committed Anny covariance, including skeleton head alignment and
/// label-based blendshape correspondence. Native .pth import is supported.
pub fn apply_anny_cached_orientation(
    data: &ModelData,
    store: &crate::assets::AssetStore,
    subtree_root: Option<&str>,
) -> Result<ModelData> {
    transaction(data, |d| store.cached_orientation(d, subtree_root))
}
/// Public SOMA-origin regression with its root=Hips convention.
pub fn regress_soma_bone_origins(
    rbf: &Tensor,
    vertices: &Tensor,
    blendshapes: &Tensor,
) -> Result<(Tensor, Tensor)> {
    ensure(
        vertices.shape.len() == 2 && vertices.shape[1] == 3,
        "vertices must be N x 3",
    )?;
    let n = vertices.shape[0];
    ensure(
        rbf.shape.len() == 2 && rbf.shape[0] >= 2 && rbf.shape[1] == n,
        "RBF matrix must have at least root and Hips",
    )?;
    ensure(
        blendshapes.shape.len() == 3 && blendshapes.shape[1..] == [n, 3],
        "morph shape mismatch",
    )?;
    vertices.validate()?;
    rbf.validate()?;
    blendshapes.validate()?;
    let j = rbf.shape[0];
    let c = blendshapes.shape[0];
    let mut heads = Tensor::zeros(vec![j, 3]);
    let mut deltas = Tensor::zeros(vec![c, j, 3]);
    for b in 0..j {
        for v in 0..n {
            let w = rbf.data[b * n + v];
            if w == 0. {
                continue;
            }
            for a in 0..3 {
                heads.data[b * 3 + a] += w * vertices.data[v * 3 + a];
            }
            for m in 0..c {
                for a in 0..3 {
                    deltas.data[(m * j + b) * 3 + a] += w * blendshapes.data[(m * n + v) * 3 + a];
                }
            }
        }
    }
    for a in 0..3 {
        heads.data[a] = heads.data[3 + a];
    }
    for m in 0..c {
        for a in 0..3 {
            deltas.data[m * j * 3 + a] = deltas.data[(m * j + 1) * 3 + a];
        }
    }
    Ok((heads, deltas))
}
