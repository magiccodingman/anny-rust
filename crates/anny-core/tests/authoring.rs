mod common;
use anny_core::{config::*, math::*, precompute::*, transforms::*, *};
use std::collections::BTreeSet;

#[test]
fn filtering_keeps_dependent_rows_and_is_transactional() -> Result<()> {
    let model = common::tiny();
    let mut d = model.data.clone();
    d.put(
        "bone_template_orientation_matrices",
        Tensor::new(vec![2, 3, 3], vec![1.; 18])?,
    );
    d.put(
        "bone_orientation_blendshapes",
        Tensor::new(vec![4, 2, 3, 3], (0..72).map(|x| x as f64).collect())?,
    );
    let result = filter_blendshapes(&d, &[true, false, true, true])?;
    assert_eq!(result.blendshape_count(), 3);
    assert_eq!(
        result.get("bone_orientation_blendshapes")?.shape,
        [3, 2, 3, 3]
    );
    assert_eq!(result.get("bone_orientation_blendshapes")?.data[18], 36.);
    assert!(filter_blendshapes(&d, &[true, true, true, false]).is_err());
    assert_eq!(d.blendshape_count(), 4);
    assert_eq!(d.get("bone_orientation_blendshapes")?.data[18], 18.);
    Ok(())
}

#[test]
fn public_pipeline_and_cached_rig_remap_are_usable() -> Result<()> {
    let model = common::tiny();
    let operations = serde_json::from_str::<Vec<Transform>>(
        r#"[{"op":"filter-blendshapes","labels":["universal:fixture"]},{"op":"compact-skinning-weights"}]"#,
    )?;
    let result = apply_pipeline(&model, &operations)?;
    assert_eq!(result.data.blendshape_count(), 1);
    assert_eq!(
        result
            .forward(&Parameters::default())?
            .get("vertices")?
            .shape,
        [1, 3, 3]
    );
    let mut d = model.data.clone();
    d.put(
        "bone_template_orientation_matrices",
        Tensor::new(
            vec![2, 3, 3],
            [1., 0., 0., 0., 1., 0., 0., 0., 1.].repeat(2),
        )?,
    );
    d.put(
        "bone_orientation_blendshapes",
        Tensor::zeros(vec![4, 2, 3, 3]),
    );
    let reduced = filter_rig(&d, &BTreeSet::from(["joint".to_string()]), None)?;
    assert_eq!(reduced.bone_count(), 1);
    assert_eq!(
        reduced.get("bone_template_orientation_matrices")?.shape,
        [1, 3, 3]
    );
    assert_eq!(
        reduced.get("bone_orientation_blendshapes")?.shape,
        [4, 1, 3, 3]
    );
    assert!(reduced
        .get("vertex_bone_weights")?
        .data
        .iter()
        .all(|x| *x == 1.));
    Ok(())
}

#[test]
fn symmetry_is_an_involution_and_cleanup_retains_binding() -> Result<()> {
    let model = common::tiny();
    let mut d = model.data.clone();
    d.put(
        "template_vertices",
        Tensor::new(vec![3, 3], vec![-1., 0., 0., 1., 0., 0., 0., 0., 1.])?,
    );
    assert_eq!(
        symmetric_vertex_indices(d.get("template_vertices")?, 0, 1e-4)?,
        vec![1, 0, 2]
    );
    d.metadata.bone_labels = vec!["root".into(), "center".into()];
    d.put(
        "vertex_bone_weights",
        Tensor::new(vec![3, 2], vec![1., 0., 0., 1., 0.4, 0.6])?,
    );
    d.put(
        "vertex_bone_indices",
        Tensor::indices(vec![3, 2], vec![0, 1, 0, 1, 0, 1]),
    );
    let result = symmetrize_skinning_weights(&d)?;
    assert_eq!(&result.get("vertex_bone_weights")?.data[..4], &[0.5; 4]);
    let clean = remove_skinning_islands(&result)?;
    clean.validate()?;
    let weights = export_weights(
        &clean,
        &serde_json::json!({"license":"CC0","weights":{"unused":[]}}),
    )?;
    assert_eq!(weights["license"], "CC0");
    assert_eq!(weights["weights"]["unused"], serde_json::json!([]));
    assert_eq!(symmetric_bone_name("LeftHand"), "RightHand");
    assert_eq!(symmetric_bone_name("finger.R"), "finger.L");
    Ok(())
}

#[test]
fn covariance_is_linear_and_falls_back_for_unweighted_bones() -> Result<()> {
    let model = common::tiny();
    let d = &model.data;
    let mut w = Tensor::zeros(vec![2, 3]);
    w.data[..3].copy_from_slice(&[0.2, 0.3, 0.5]);
    let rotations = Tensor::new(
        vec![2, 3, 3],
        [1., 0., 0., 0., 1., 0., 0., 0., 1.].repeat(2),
    )?;
    let inputs = OrientationInputs {
        template_vertices: d.get("template_vertices")?,
        blendshapes: d.get("blendshapes")?,
        template_origins: d.get("template_bone_heads")?,
        origin_blendshapes: d.get("bone_heads_blendshapes")?,
        vertex_weights: &w,
        reference_vertices: d.get("template_vertices")?,
        reference_orientations: &rotations,
        reference_origins: d.get("template_bone_heads")?,
        parents: &d.metadata.bone_parents,
        template_tails: d.arrays.get("template_bone_tails"),
        tail_blendshapes: d.arrays.get("bone_tails_blendshapes"),
        reference_tails: d.arrays.get("template_bone_tails"),
    };
    for target in [AimTarget::Tail, AimTarget::Children] {
        for weight in [0., 0.5] {
            let cache = compute_cached_orientation_data(&inputs, weight, target)?;
            assert_eq!(&cache.template.data[9..], &rotations.data[9..]);
            for b in 0..d.blendshape_count() {
                assert!(cache.blendshapes.data[(b * 2 + 1) * 9..(b * 2 + 2) * 9]
                    .iter()
                    .all(|&v| v == 0.));
            }
            let combined = mat3(&cache.template.data[..9])
                + 0.2 * mat3(&cache.blendshapes.data[2 * 2 * 9..2 * 2 * 9 + 9]);
            assert!(combined.iter().all(|x| x.is_finite()));
        }
    }
    let mut nondegenerate = model.clone();
    nondegenerate
        .data
        .arrays
        .get_mut("template_vertices")
        .unwrap()
        .data[8] = 1.2;
    let cached = cache_model_orientations(
        &nondegenerate,
        &Parameters::default(),
        &OrientationOptions {
            align_root_with_pelvis: false,
            ..Default::default()
        },
    )?;
    assert_eq!(cached.rig.bone_orientation, BoneOrientation::Cached);
    cached.forward(&Parameters::default())?;
    Ok(())
}

#[test]
#[ignore = "real native preprocessing/reference check; not part of the fast suite"]
fn canonical_orientation_caches_match_committed_references() -> Result<()> {
    let store = assets::AssetStore::new(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data"),
    );
    for name in ["anny", "soma"] {
        eprintln!("Preparing {name} natively");
        let actual = if name == "anny" {
            precompute_anny(&store, &OrientationOptions::default())?
        } else {
            precompute_soma(&store, 0.01)?
        };
        let expected = store.converted(&format!("cached/{name}.pth"))?;
        for key in ["bone_labels", "blendshape_labels"] {
            assert_eq!(
                actual.payload()?[key],
                expected.payload()?[key],
                "{name} {key}"
            );
        }
        for key in [
            "bone_template_orientation_matrices",
            "bone_orientation_blendshapes",
            "reference_bone_orientations",
        ] {
            let a = actual.payload_tensor(key)?;
            let e = expected.payload_tensor(key)?;
            assert_eq!(a.shape, e.shape, "{name}/{key}");
            let max = a
                .data
                .iter()
                .zip(&e.data)
                .map(|(a, e)| (a - e).abs())
                .fold(0., f64::max);
            eprintln!("{name}/{key} max absolute difference {max:e}");
            assert!(max < 1e-6, "{name}/{key}: {max}");
        }
    }
    Ok(())
}

#[test]
#[ignore = "real skin-weight preprocessing smoke"]
fn raw_skin_weight_cleanup_is_operational() -> Result<()> {
    let store = assets::AssetStore::new(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data"),
    );
    let weights = compute_cleaned_weights(&store)?;
    let w = weights["weights"].as_object().unwrap();
    assert!(w.len() >= 100);
    assert!(w.values().any(|v| v.as_array().unwrap().len() > 100));
    let expected: serde_json::Value = serde_json::from_slice(&std::fs::read(
        store.root.join("mpfb2/rigs/standard/weights.default.json"),
    )?)?;
    let mut max = 0.0f64;
    // Original detached/unweighted helper vertices use the port's documented
    // root-bound fallback. Compare every actually weighted reference entry.
    for (name, entries) in expected["weights"].as_object().unwrap() {
        let actual: std::collections::BTreeMap<u64, f64> = w
            .get(name)
            .and_then(|v| v.as_array())
            .unwrap()
            .iter()
            .map(|e| (e[0].as_u64().unwrap(), e[1].as_f64().unwrap()))
            .collect();
        for e in entries.as_array().unwrap() {
            let v = e[0].as_u64().unwrap();
            let a = actual.get(&v).copied().unwrap_or(0.);
            max = max.max((a - e[1].as_f64().unwrap()).abs());
        }
    }
    eprintln!("raw skin-weight recomputation max reference-entry difference {max:e}");
    assert!(max < 1e-6);
    Ok(())
}

#[test]
fn signed_weight_interpolation_and_runtime_procrustes_retopology() -> Result<()> {
    let m = common::tiny();
    let refs = Tensor::indices(vec![1, 2], vec![0, 2]);
    let weights = Tensor::new(vec![1, 2], vec![2., -1.])?;
    let skin = interpolate_skinning_weights(&m.data, &refs, &weights, true)?;
    assert_eq!(skin.indices.data, vec![0., 1.]);
    assert_eq!(skin.weights.data, vec![2., -1.]);
    let cancel = Tensor::indices(vec![1, 2], vec![0, 0]);
    let cancel_w = Tensor::new(vec![1, 2], vec![1., -1.])?;
    assert!(interpolate_skinning_weights(&m.data, &cancel, &cancel_w, true).is_err());
    assert!(
        interpolate_skinning_weights(&m.data, &cancel, &cancel_w, false)?
            .weights
            .data
            .iter()
            .all(|&x| x == 0.)
    );
    let data = apply_procrustes_orientation(&m.data)?;
    let vertices = data.get("template_vertices")?;
    let ids = Tensor::indices(vec![3, 1], vec![0, 1, 2]);
    let weights = Tensor::new(vec![3, 1], vec![1.; 3])?;
    let target =
        apply_procrustes_retopology(&data, vertices, data.get("faces")?, &ids, &weights, None)?;
    let a = anny_core::model::rest_model(
        &data,
        &RigConfig {
            bone_orientation: BoneOrientation::Procrustes,
            ..RigConfig::parse("makehuman")?
        },
        &Tensor::zeros(vec![1, 4]),
    )?;
    let b = anny_core::model::rest_model(
        &target,
        &RigConfig {
            bone_orientation: BoneOrientation::Procrustes,
            ..RigConfig::parse("makehuman")?
        },
        &Tensor::zeros(vec![1, 4]),
    )?;
    for (a, b) in a
        .get("rest_vertices")?
        .data
        .iter()
        .zip(&b.get("rest_vertices")?.data)
    {
        assert!((a - b).abs() < 1e-10);
    }
    assert_eq!(target.get("bone_vertex_indices")?.shape[0], 2);
    Ok(())
}

#[test]
fn portable_secondary_requests_reject_unknown_options() -> Result<()> {
    let m = common::tiny();
    let response = operations::execute_json(&m, r#"{"operation":"pose-convert","mode":"world"}"#)?;
    assert!(response.contains("pose_parameters"));
    assert!(operations::execute_json(
        &m,
        r#"{"operation":"pose-convert","mode":"world","ignored":true}"#
    )
    .is_err());
    assert!(operations::execute_json(&m, r#"{"operation":"imaginary"}"#).is_err());
    let keypoints = tools::KeypointsRegressor {
        labels: vec!["point".into()],
        weights: Tensor::new(vec![1, 3], vec![1., 0., 0.])?,
        indices: None,
    };
    let request = operations::Request::Keypoints {
        parameters: Parameters::default(),
        regressor: keypoints,
    };
    assert_eq!(
        operations::execute(&m, &request)?["labels"],
        serde_json::json!(["point"])
    );
    Ok(())
}
