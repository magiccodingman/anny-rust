//! Native covariance/rig preprocessing, translated from NAVER Anny's
//! scripts/precompute_rig_caches.py. Copyright (C) 2025 NAVER Corp.
//! SPDX-License-Identifier: Apache-2.0
use crate::{
    assets::AssetStore,
    config::*,
    ensure,
    math::*,
    model::{Anny, ModelData, Parameters},
    tensor::Archive,
    Error, Result, Tensor,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Weighting {
    Skinning,
    #[default]
    SkinningSquared,
    PrincipalSquared,
}
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AimTarget {
    #[default]
    Tail,
    Children,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OrientationOptions {
    pub weighting: Weighting,
    pub aim_weight: f64,
    pub aim_target: AimTarget,
    pub align_root_with_pelvis: bool,
}
impl Default for OrientationOptions {
    fn default() -> Self {
        Self {
            weighting: Weighting::SkinningSquared,
            aim_weight: 0.5,
            aim_target: AimTarget::Tail,
            align_root_with_pelvis: true,
        }
    }
}
/// Borrowed inputs to the source's linear covariance construction. No interpreter,
/// GPU, automatic differentiation, or pre-existing orientation cache is needed.
pub struct OrientationInputs<'a> {
    pub template_vertices: &'a Tensor,
    pub blendshapes: &'a Tensor,
    pub template_origins: &'a Tensor,
    pub origin_blendshapes: &'a Tensor,
    pub vertex_weights: &'a Tensor,
    pub reference_vertices: &'a Tensor,
    pub reference_orientations: &'a Tensor,
    pub reference_origins: &'a Tensor,
    pub parents: &'a [i32],
    pub template_tails: Option<&'a Tensor>,
    pub tail_blendshapes: Option<&'a Tensor>,
    pub reference_tails: Option<&'a Tensor>,
}
#[derive(Clone, Debug)]
pub struct OrientationCache {
    pub template: Tensor,
    pub blendshapes: Tensor,
}
impl OrientationCache {
    pub fn apply(&self, data: &mut ModelData) -> Result<()> {
        self.template
            .expect_shape(&[data.bone_count(), 3, 3], "cached covariance")?;
        self.blendshapes.expect_shape(
            &[data.blendshape_count(), data.bone_count(), 3, 3],
            "cached covariance blendshapes",
        )?;
        data.put("bone_template_orientation_matrices", self.template.clone());
        data.put("bone_orientation_blendshapes", self.blendshapes.clone());
        Ok(())
    }
}
pub fn compute_cached_orientation_data(
    input: &OrientationInputs<'_>,
    aim_weight: f64,
    aim_target: AimTarget,
) -> Result<OrientationCache> {
    let n = input.template_vertices.shape.first().copied().unwrap_or(0);
    let c = input.blendshapes.shape.first().copied().unwrap_or(0);
    let j = input.parents.len();
    ensure(
        n > 0 && j > 0 && aim_weight.is_finite() && aim_weight >= 0.,
        "invalid covariance dimensions or aim weight",
    )?;
    propagation_order(input.parents)?;
    for (t, shape, name) in [
        (input.template_vertices, vec![n, 3], "template vertices"),
        (input.blendshapes, vec![c, n, 3], "blendshapes"),
        (input.template_origins, vec![j, 3], "template origins"),
        (
            input.origin_blendshapes,
            vec![c, j, 3],
            "origin blendshapes",
        ),
        (input.vertex_weights, vec![j, n], "orientation weights"),
        (input.reference_vertices, vec![n, 3], "reference vertices"),
        (
            input.reference_orientations,
            vec![j, 3, 3],
            "reference orientations",
        ),
        (input.reference_origins, vec![j, 3], "reference origins"),
    ] {
        t.expect_shape(&shape, name)?;
    }
    ensure(
        input.vertex_weights.data.iter().all(|&w| w >= 0.),
        "orientation weights must be nonnegative",
    )?;
    if aim_weight > 0. && matches!(aim_target, AimTarget::Tail) {
        for (t, shape) in [
            (input.template_tails, vec![j, 3]),
            (input.tail_blendshapes, vec![c, j, 3]),
            (input.reference_tails, vec![j, 3]),
        ] {
            t.ok_or_else(|| Error::Invalid("tail aiming requires authored tail data".into()))?
                .expect_shape(&shape, "tail data")?;
        }
    }
    let mut base = Tensor::zeros(vec![j, 3, 3]);
    let mut shapes = Tensor::zeros(vec![c, j, 3, 3]);
    let v3 = |t: &Tensor, i: usize| vec3(&t.data[i * 3..i * 3 + 3]);
    for bone in 0..j {
        let rotation = mat3(&input.reference_orientations.data[bone * 9..bone * 9 + 9]);
        ensure(
            (rotation.transpose() * rotation - Mat3::identity()).norm() < 1e-5
                && (rotation.determinant() - 1.).abs() < 1e-5,
            "reference frame is not a rotation",
        )?;
        let samples: Vec<_> = input.vertex_weights.data[bone * n..(bone + 1) * n]
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, w)| *w > 0.)
            .collect();
        if samples.is_empty() {
            write3(&rotation, &mut base.data[bone * 9..bone * 9 + 9]);
            continue;
        }
        let ref_origin = v3(input.reference_origins, bone);
        let origin = v3(input.template_origins, bone);
        let norm2: f64 = samples
            .iter()
            .map(|&(i, w)| (w * (v3(input.reference_vertices, i) - ref_origin)).norm_squared())
            .sum();
        ensure(
            norm2 > 0. && norm2.is_finite(),
            "weighted reference samples collapse to their bone origin",
        )?;
        let scaling = 1. / norm2.sqrt();
        let refs: Vec<_> = samples
            .iter()
            .map(|&(i, w)| {
                (
                    i,
                    w,
                    rotation.transpose()
                        * (scaling * (v3(input.reference_vertices, i) - ref_origin)),
                )
            })
            .collect();
        let mut m = Mat3::zeros();
        let mut bs = vec![Mat3::zeros(); c];
        for &(i, w, r) in &refs {
            m += w * (scaling * (v3(input.template_vertices, i) - origin)) * r.transpose();
        }
        for (b, delta) in bs.iter_mut().enumerate() {
            let center = v3(input.origin_blendshapes, b * j + bone);
            for &(i, w, r) in &refs {
                *delta +=
                    w * (scaling * (v3(input.blendshapes, b * n + i) - center)) * r.transpose();
            }
        }
        if aim_weight > 0. {
            let targets: Vec<usize> = match aim_target {
                AimTarget::Tail => vec![bone],
                AimTarget::Children => input
                    .parents
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| **p == bone as i32)
                    .map(|(i, _)| i)
                    .collect(),
            };
            let mut aim = Mat3::zeros();
            let mut aim_bs = vec![Mat3::zeros(); c];
            let mut valid = false;
            for child in targets {
                let (ref_target, template_target, offsets) = match aim_target {
                    AimTarget::Tail => (
                        v3(input.reference_tails.unwrap(), child),
                        v3(input.template_tails.unwrap(), child),
                        input.tail_blendshapes.unwrap(),
                    ),
                    AimTarget::Children => (
                        v3(input.reference_origins, child),
                        v3(input.template_origins, child),
                        input.origin_blendshapes,
                    ),
                };
                let offset = ref_target - ref_origin;
                if matches!(aim_target, AimTarget::Tail) && offset.norm() < 1e-9 {
                    continue;
                }
                valid = true;
                let local = rotation.transpose() * offset;
                aim += (template_target - origin) * local.transpose();
                for (b, delta) in aim_bs.iter_mut().enumerate() {
                    *delta += (v3(offsets, b * j + child)
                        - v3(input.origin_blendshapes, b * j + bone))
                        * local.transpose();
                }
            }
            if valid {
                let vertex_scale = m.norm() + 1e-12;
                let aim_scale = aim.norm() + 1e-12;
                m = m / vertex_scale + aim_weight * aim / aim_scale;
                for (b, delta) in bs.iter_mut().enumerate() {
                    *delta = *delta / vertex_scale + aim_weight * aim_bs[b] / aim_scale;
                }
            }
        }
        write3(&m, &mut base.data[bone * 9..bone * 9 + 9]);
        for (b, delta) in bs.iter().enumerate() {
            write3(
                delta,
                &mut shapes.data[(b * j + bone) * 9..(b * j + bone + 1) * 9],
            );
        }
    }
    base.validate()?;
    shapes.validate()?;
    Ok(OrientationCache {
        template: base,
        blendshapes: shapes,
    })
}
pub fn bone_vertex_weights(data: &ModelData, strategy: Weighting) -> Result<Tensor> {
    data.validate()?;
    let n = data.vertex_count();
    let j = data.bone_count();
    let ids = data
        .get("vertex_bone_indices")?
        .checked_indices(j, "weights")?;
    let w = data.get("vertex_bone_weights")?;
    let k = w.shape[1];
    let mut dense = Tensor::zeros(vec![j, n]);
    for v in 0..n {
        if matches!(strategy, Weighting::PrincipalSquared) {
            let s = (0..k)
                .max_by(|&a, &b| {
                    w.data[v * k + a]
                        .total_cmp(&w.data[v * k + b])
                        .then(b.cmp(&a))
                })
                .unwrap();
            dense.data[ids[v * k + s] * n + v] = w.data[v * k + s].powi(2);
        } else {
            for s in 0..k {
                dense.data[ids[v * k + s] * n + v] += w.data[v * k + s];
            }
        }
    }
    if matches!(strategy, Weighting::SkinningSquared) {
        for w in &mut dense.data {
            *w *= *w;
        }
    }
    Ok(dense)
}
fn extract_reference(
    model: &Anny,
    parameters: &Parameters,
) -> Result<(Tensor, Tensor, Tensor, Option<Tensor>)> {
    let rest = model.rest_model(&model.coefficients(parameters)?)?;
    let positions = rest.get("rest_vertices")?;
    ensure(
        positions.shape[0] == 1,
        "orientation reference must contain one shape",
    )?;
    let poses = rest.get("rest_bone_poses")?;
    let mut rotations = Tensor::zeros(vec![model.data.bone_count(), 3, 3]);
    let mut origins = Tensor::zeros(vec![model.data.bone_count(), 3]);
    for (b, m) in poses.data.chunks_exact(16).enumerate() {
        let m = mat4(m);
        write3(&rotation(&m), &mut rotations.data[b * 9..b * 9 + 9]);
        origins.data[b * 3..b * 3 + 3].copy_from_slice(translation(&m).as_slice());
    }
    let vertices = Tensor::new(vec![positions.shape[1], 3], positions.data.clone())?;
    let tails = rest
        .arrays
        .get("rest_bone_tails")
        .map(|t| Tensor::new(vec![t.shape[1], 3], t.data.clone()))
        .transpose()?;
    Ok((vertices, rotations, origins, tails))
}
/// Convert any supported model to cached-covariance orientation using an explicit
/// reference shape. The caller controls aim settings and root realignment.
pub fn cache_model_orientations(
    model: &Anny,
    reference: &Parameters,
    options: &OrientationOptions,
) -> Result<Anny> {
    let (vertices, rotations, origins, tails) = extract_reference(model, reference)?;
    let weights = bone_vertex_weights(&model.data, options.weighting)?;
    let d = &model.data;
    let cache = compute_cached_orientation_data(
        &OrientationInputs {
            template_vertices: d.get("template_vertices")?,
            blendshapes: d.get("blendshapes")?,
            template_origins: d.get("template_bone_heads")?,
            origin_blendshapes: d.get("bone_heads_blendshapes")?,
            vertex_weights: &weights,
            reference_vertices: &vertices,
            reference_orientations: &rotations,
            reference_origins: &origins,
            parents: &d.metadata.bone_parents,
            template_tails: d.arrays.get("template_bone_tails"),
            tail_blendshapes: d.arrays.get("bone_tails_blendshapes"),
            reference_tails: tails.as_ref(),
        },
        options.aim_weight,
        options.aim_target,
    )?;
    let mut data = d.clone();
    cache.apply(&mut data)?;
    data.put("reference_bone_orientations", rotations);
    if options.align_root_with_pelvis {
        align_root(&mut data)?;
    }
    data.remove(&[
        "bone_nonzeroweight_mask",
        "bone_vertex_indices",
        "bone_vertex_weights",
        "template_bone_vertices",
        "template_bone_tails",
        "bone_tails_blendshapes",
        "bone_rolls_rotmat",
    ]);
    let mut config = model.config.clone();
    let mut rig = model.rig.clone();
    rig.bone_orientation = BoneOrientation::Cached;
    config.rig = RigSpec::Config(rig);
    Anny::from_model_data(data, config)
}
fn align_root(data: &mut ModelData) -> Result<()> {
    let left = data
        .metadata
        .bone_labels
        .iter()
        .position(|n| n == "pelvis.L")
        .ok_or_else(|| {
            Error::Invalid("root realignment needs pelvis.L/R; disable it for other rigs".into())
        })?;
    let right = data
        .metadata
        .bone_labels
        .iter()
        .position(|n| n == "pelvis.R")
        .ok_or_else(|| Error::Invalid("root realignment needs pelvis.R".into()))?;
    let heads = data.get("template_bone_heads")?;
    ensure(
        heads.data[left * 3..left * 3 + 3] == heads.data[right * 3..right * 3 + 3],
        "pelvis heads differ",
    )?;
    let w = data.get("vertex_bone_weights")?;
    let ids = data.get("vertex_bone_indices")?;
    ensure(
        w.data
            .iter()
            .zip(&ids.data)
            .all(|(&w, &i)| i != 0. || w == 0.),
        "cannot move a weighted root",
    )?;
    let j = data.bone_count();
    let c = data.blendshape_count();
    let mut heads = heads.clone();
    let v = heads.data[left * 3..left * 3 + 3].to_vec();
    heads.data[..3].copy_from_slice(&v);
    let mut bs = data.get("bone_heads_blendshapes")?.clone();
    for b in 0..c {
        ensure(
            bs.data[(b * j + left) * 3..(b * j + left + 1) * 3]
                == bs.data[(b * j + right) * 3..(b * j + right + 1) * 3],
            "pelvis origin morphs differ",
        )?;
        let v = bs.data[(b * j + left) * 3..(b * j + left + 1) * 3].to_vec();
        bs.data[b * j * 3..b * j * 3 + 3].copy_from_slice(&v);
    }
    data.put("template_bone_heads", heads);
    data.put("bone_heads_blendshapes", bs);
    Ok(())
}
fn cache_archive(data: &ModelData, mut payload: Value, keys: &[&str]) -> Result<Archive> {
    let mut out = Archive::default();
    payload["bone_labels"] = json!(data.metadata.bone_labels);
    payload["blendshape_labels"] = json!(data.metadata.blendshape_labels);
    for &key in keys {
        out.tensors.insert(key.into(), data.get(key)?.clone());
        payload[key] = json!({"__tensor__":key});
    }
    out.metadata
        .insert("anny_port_payload".into(), serde_json::to_string(&payload)?);
    out.metadata
        .insert("anny_rust_upstream".into(), crate::UPSTREAM_REVISION.into());
    Ok(out)
}
/// Rebuild the canonical Anny orientation cache from repository assets, matching
/// the upstream authored reference age, weighting, aiming, and pelvis convention.
pub fn precompute_anny(store: &AssetStore, options: &OrientationOptions) -> Result<Archive> {
    let config = AnnyConfig {
        rig: RigSpec::Name("makehuman-notongue-nobreasts-nofacialexpression-pruned".into()),
        local_changes: Selection::all(),
        ..Default::default()
    };
    let model = store.build(&config)?;
    let reference = Parameters {
        phenotype_kwargs: json!({"age":2./3.}),
        ..Default::default()
    };
    let model = cache_model_orientations(&model, &reference, options)?;
    cache_archive(
        &model.data,
        json!({"bone_orientation_weighting_strategy":options.weighting,"bone_orientation_centering_strategy":"head","aim_weight":options.aim_weight,"aim_target":options.aim_target}),
        &[
            "template_bone_heads",
            "bone_heads_blendshapes",
            "reference_bone_orientations",
            "bone_template_orientation_matrices",
            "bone_orientation_blendshapes",
        ],
    )
}
/// Rebuild SOMA covariance using its committed bind shape/rig. This does not
/// download SOMA-X or regenerate the separately authored RBF rig asset itself.
pub fn precompute_soma(store: &AssetStore, threshold: f64) -> Result<Archive> {
    ensure(
        threshold.is_finite() && (0.0..1.0).contains(&threshold),
        "SOMA threshold must be in [0,1)",
    )?;
    // Build only the retopologized geometry with tail orientation. Neither the
    // existing Anny covariance nor the existing SOMA covariance is required.
    let config = AnnyConfig {
        local_changes: Selection::all(),
        ..Default::default()
    };
    let mut d = store.build_alternative(
        &config,
        &RigConfig::parse("makehuman")?,
        &TopologyConfig::parse("soma")?,
    )?;
    let rig = store.converted("soma/soma_rig.pt")?;
    let payload = rig.payload()?;
    let labels: Vec<String> = serde_json::from_value(payload["bone_labels"].clone())?;
    let parents: Vec<i32> = serde_json::from_value(payload["bone_parents"].clone())?;
    ensure(
        labels.len() == parents.len() && labels.len() > 1,
        "invalid SOMA bone metadata",
    )?;
    propagation_order(&parents)?;
    let n = d.vertex_count();
    let j = labels.len();
    let c = d.blendshape_count();
    let rbf = rig.payload_tensor("sparse_rbf_matrix")?;
    rbf.expect_shape(&[j, n], "SOMA origin regression")?;
    let template = d.get("template_vertices")?;
    let morphs = d.get("blendshapes")?;
    let mut heads = Tensor::zeros(vec![j, 3]);
    let mut head_shapes = Tensor::zeros(vec![c, j, 3]);
    for b in 0..j {
        for v in 0..n {
            let w = rbf.data[b * n + v];
            if w == 0. {
                continue;
            }
            for a in 0..3 {
                heads.data[b * 3 + a] += w * template.data[v * 3 + a];
            }
            for morph in 0..c {
                for a in 0..3 {
                    head_shapes.data[(morph * j + b) * 3 + a] +=
                        w * morphs.data[(morph * n + v) * 3 + a];
                }
            }
        }
    }
    for a in 0..3 {
        heads.data[a] = heads.data[3 + a];
    }
    for b in 0..c {
        for a in 0..3 {
            head_shapes.data[b * j * 3 + a] = head_shapes.data[(b * j + 1) * 3 + a];
        }
    }
    let tpose = rig.payload_tensor("t_pose_world")?;
    tpose.expect_shape(&[j, 4, 4], "SOMA t-pose")?;
    let mut reference = Tensor::zeros(vec![j, 3, 3]);
    for (b, t) in tpose.data.chunks_exact(16).enumerate() {
        write3(&rotation(&mat4(t)), &mut reference.data[b * 9..b * 9 + 9]);
    }
    d.metadata.bone_labels = labels;
    d.metadata.bone_parents = parents;
    d.put("template_bone_heads", heads);
    d.put("bone_heads_blendshapes", head_shapes);
    d.put("reference_bone_orientations", reference);
    let skin = rig.payload_tensor("skinning_weights")?;
    skin.expect_shape(&[n, j], "SOMA skinning")?;
    let bind = rig.payload_tensor("bind_world_transforms")?;
    bind.expect_shape(&[j, 4, 4], "SOMA bind transforms")?;
    let mut rotations = Tensor::zeros(vec![j, 3, 3]);
    let mut origins = Tensor::zeros(vec![j, 3]);
    let mut weights = Tensor::zeros(vec![j, n]);
    for b in 0..j {
        let transform = mat4(&bind.data[b * 16..b * 16 + 16]);
        write3(&rotation(&transform), &mut rotations.data[b * 9..b * 9 + 9]);
        origins.data[b * 3..b * 3 + 3].copy_from_slice(translation(&transform).as_slice());
        for v in 0..n {
            weights.data[b * n + v] = f64::from(skin.data[v * j + b] > threshold);
        }
    }
    let cache = compute_cached_orientation_data(
        &OrientationInputs {
            template_vertices: d.get("template_vertices")?,
            blendshapes: d.get("blendshapes")?,
            template_origins: d.get("template_bone_heads")?,
            origin_blendshapes: d.get("bone_heads_blendshapes")?,
            vertex_weights: &weights,
            reference_vertices: rig.payload_tensor("bind_shape")?,
            reference_orientations: &rotations,
            reference_origins: &origins,
            parents: &d.metadata.bone_parents,
            template_tails: None,
            tail_blendshapes: None,
            reference_tails: None,
        },
        0.,
        AimTarget::Children,
    )?;
    let mut out = d;
    cache.apply(&mut out)?;
    cache_archive(
        &out,
        json!({"bone_orientation_weighting_strategy":"binary_threshold","bone_orientation_centering_strategy":"head","weight_threshold":threshold}),
        &[
            "reference_bone_orientations",
            "bone_template_orientation_matrices",
            "bone_orientation_blendshapes",
        ],
    )
}

/// Re-run the default skin-weight cleanup from raw MakeHuman weights. Writes are
/// left to the caller; source assets are never modified by this function.
pub fn compute_cleaned_weights(store: &AssetStore) -> Result<Value> {
    let source = store
        .root
        .join("mpfb2/rigs/standard/weights.makehuman.json");
    let header: Value = serde_json::from_slice(&std::fs::read(&source)?)?;
    let mut rig = RigConfig::parse("makehuman")?;
    rig.weights_filename = Some(source.canonicalize()?.to_string_lossy().into_owned());
    let config = AnnyConfig {
        rig: RigSpec::Config(rig),
        topology: TopologySpec::Config(TopologyConfig {
            nudity_edits: true,
            remove_unattached_vertices: false,
            triangulate_faces: false,
            ..Default::default()
        }),
        ..Default::default()
    };
    let model = store.build(&config)?;
    let d = crate::transforms::symmetrize_skinning_weights(&model.data)?;
    let d = crate::transforms::remove_skinning_islands(&d)?;
    crate::transforms::export_weights(&d, &header)
}
