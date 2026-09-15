mod common;
use anny_core::{math::*, motion::*, numpy::*, PoseParameterization, Tensor};
use serde_json::json;
use std::{
    collections::BTreeMap,
    io::{Cursor, Write},
};

fn npy(dtype: &str, shape: &str, fortran: bool, data: &[u8]) -> Vec<u8> {
    let mut header = format!(
        "{{'descr': '{dtype}', 'fortran_order': {}, 'shape': {shape}, }}",
        if fortran { "True" } else { "False" }
    );
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(data);
    out
}
fn zip(files: &[(&str, Vec<u8>)]) -> Vec<u8> {
    let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
    for (name, bytes) in files {
        writer
            .start_file(
                *name,
                zip::write::FileOptions::default()
                    .compression_method(zip::CompressionMethod::Deflated),
            )
            .unwrap();
        writer.write_all(bytes).unwrap();
    }
    writer.finish().unwrap().into_inner()
}
#[test]
fn numpy_endian_fortran_metadata_and_hostile_inputs() {
    let values: [f32; 6] = [1., 4., 2., 5., 3., 6.];
    let bytes = values
        .iter()
        .flat_map(|v| v.to_be_bytes())
        .collect::<Vec<_>>();
    let a = npy(">f4", "(2,3)", true, &bytes);
    let result = read_npy(&a, Limits::default()).unwrap();
    assert_eq!(result.numeric().unwrap().data, vec![1., 2., 3., 4., 5., 6.]);
    let text = "neutral"
        .chars()
        .flat_map(|c| (c as u32).to_le_bytes())
        .collect::<Vec<_>>();
    assert_eq!(
        read_npy(&npy("<U7", "()", false, &text), Limits::default())
            .unwrap()
            .scalar_text()
            .unwrap(),
        "neutral"
    );
    assert_eq!(
        read_npy(&npy("|S4", "()", false, b"male"), Limits::default())
            .unwrap()
            .scalar_text()
            .unwrap(),
        "male"
    );
    for bad in [
        npy("|O8", "(1,)", false, &[0; 8]),
        npy("<f8", "(1,)", false, &f64::NAN.to_le_bytes()),
        npy("<f8", "(99999999999999999999999,)", false, &[]),
    ] {
        assert!(read_npy(&bad, Limits::default()).is_err());
    }
    assert!(read_npy(&a[..a.len() - 1], Limits::default()).is_err());
    assert!(read_npy(
        &a,
        Limits {
            max_numeric_values: 2,
            ..Default::default()
        }
    )
    .is_err());
    assert!(read_npz(&zip(&[("../evil.npy", a.clone())]), Limits::default()).is_err());
    assert!(read_npz(
        &zip(&[("x.npy", a.clone()), ("x.npy", a.clone())]),
        Limits::default()
    )
    .is_err());
    assert!(read_npz(
        &zip(&[("x.npy", a.clone())]),
        Limits {
            max_uncompressed_bytes: 10,
            ..Default::default()
        }
    )
    .is_err());
    assert_eq!(
        read_npz(&zip(&[("x.npy", a)]), Limits::default()).unwrap()["x"]
            .numeric()
            .unwrap()
            .data
            .len(),
        6
    );
}
#[test]
fn amass_arrays_not_silently_used_as_anny_poses() {
    let files = vec![
        ("root_orient.npy", npy("<f4", "(2,3)", false, &[0; 24])),
        ("pose_body.npy", npy("<f4", "(2,63)", false, &[0; 504])),
        ("betas.npy", npy("<f8", "(10,)", false, &[0; 80])),
        (
            "mocap_frame_rate.npy",
            npy("<f8", "()", false, &60f64.to_le_bytes()),
        ),
        ("gender.npy", npy("|S7", "()", false, b"neutral")),
    ];
    let sequence = AmassSequence::from_npz(&zip(&files)).unwrap();
    assert_eq!(sequence.frame_count(), 2);
    assert_eq!(sequence.fps, Some(60.));
    assert_eq!(sequence.gender.as_deref(), Some("neutral"));
    assert_eq!(sequence.pose_hand.shape, vec![2, 90]);
    assert!(sequence.pose_hand.data.iter().all(|x| *x == 0.));
    assert!(AmassSequence::from_arrays(&BTreeMap::from([(
        "poses".into(),
        NpyValue::Numeric(Tensor::zeros(vec![2, 156]))
    )]))
    .is_err());
}
#[test]
fn clips_resample_transfer_and_export_in_both_precisions() {
    let model = common::tiny();
    let vectors = Tensor::new(
        vec![2, 2, 3],
        vec![
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            std::f64::consts::FRAC_PI_2,
            0.,
            0.,
            0.,
        ],
    )
    .unwrap();
    let translations = Tensor::new(vec![2, 3], vec![0., 0., 0., 2., 0., 0.]).unwrap();
    let clip = PoseClip::from_rotvecs(
        &model,
        &vectors,
        &translations,
        1.,
        PoseParameterization::LocalBone,
    )
    .unwrap();
    let interp = clip.resample(&model, 2.).unwrap();
    assert_eq!(interp.frames.len(), 3);
    let mid = anny_core::model::parse_pose(
        &interp.frames[1].pose_parameters,
        &model.data.metadata.bone_labels,
    )
    .unwrap();
    assert!((mid.data[3] - 1.).abs() < 1e-12);
    assert!((mid.data[0] - 2f64.sqrt() / 2.).abs() < 1e-12);
    let transferred = interp
        .transfer(&model, &model, PoseParameterization::World)
        .unwrap();
    for i in 0..3 {
        let a = model.forward(&interp.frame_parameters(i).unwrap()).unwrap();
        let b = model
            .forward(&transferred.frame_parameters(i).unwrap())
            .unwrap();
        for (a, b) in a
            .get("vertices")
            .unwrap()
            .data
            .iter()
            .zip(&b.get("vertices").unwrap().data)
        {
            assert!((a - b).abs() < 1e-9);
        }
    }
    // Both precisions must export the same motion, and the motion must actually move. A session that
    // never applied its per-frame pose would still produce one animation of the right shape, so the
    // values are what has to be asserted, not the document structure.
    let scenes: Vec<anny_core::scene::Scene> = [Precision::F32, Precision::F64]
        .into_iter()
        .map(|mode| interp.to_scene(&model, mode).unwrap())
        .collect();
    for scene in &scenes {
        assert_eq!(scene.animations.len(), 1);
        let animation = &scene.animations[0];
        assert_eq!(animation.times.len(), interp.frames.len());
        assert_eq!(animation.bone_poses.len(), interp.frames.len());
        assert_ne!(
            animation.bone_poses[0], animation.bone_poses[1],
            "the exported animation does not move between frames"
        );
        let glb = scene.to_glb().unwrap();
        assert_eq!(&glb[..4], b"glTF");
        let len = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
        let doc: serde_json::Value = serde_json::from_slice(&glb[20..20 + len]).unwrap();
        assert_eq!(doc["animations"].as_array().unwrap().len(), 1);
    }
    for (single_pose, double_pose) in scenes[0].animations[0]
        .bone_poses
        .iter()
        .zip(&scenes[1].animations[0].bone_poses)
    {
        assert_eq!(single_pose.len(), double_pose.len());
        for (single, double) in single_pose.iter().zip(double_pose) {
            for (single, double) in single.iter().zip(double.iter()) {
                assert!(
                    (single - double).abs() < 1e-6,
                    "the f32 export disagrees with the f64 one"
                );
            }
        }
    }
    let request = json!({"operation":"motion-resample","clip":clip,"fps":2.});
    assert_eq!(
        anny_core::operations::execute_json(&model, &request.to_string()).unwrap(),
        serde_json::to_string(&interp).unwrap()
    );
    let mut bad = clip.clone();
    bad.frames[1].time = 0.;
    assert!(bad.validate(&model).is_err());
    bad = clip.clone();
    bad.frames[1].pose_parameters =
        json!({"missing-bone":[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]});
    assert!(bad.validate(&model).is_err());
    assert!(clip.resample(&model, f64::INFINITY).is_err());
}
#[test]
fn landmarks_recover_known_similarity_and_reject_degenerate_pairs() {
    use anny_core::fitting::Landmarks;
    let points = vec![[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
    let rotation = rotvec(&Vec3::new(0.2, -0.4, 0.1));
    let transform = rigid(&(rotation * 2.), &Vec3::new(4., -2., 1.));
    let landmarks = Landmarks {
        target_points: points.clone(),
        model_points: points
            .iter()
            .map(|p| point(&transform, &vec3(p)).into())
            .collect(),
        weights: vec![1., 2., 0.5, 1.],
        estimate_scale: true,
    };
    let alignment = landmarks.align().unwrap();
    assert!((alignment.scale - 2.).abs() < 1e-10);
    assert!(alignment.weighted_rms < 1e-10);
    for (i, row) in alignment.target_transform.iter().enumerate() {
        for (j, x) in row.iter().enumerate() {
            assert!((x - transform[(i, j)]).abs() < 1e-10);
        }
    }
    let line = Landmarks {
        target_points: vec![[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]],
        model_points: vec![[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]],
        weights: vec![],
        estimate_scale: false,
    };
    assert!(line.align().is_err());
    let mut bad = landmarks.clone();
    bad.weights[0] = -1.;
    assert!(bad.align().is_err());
    bad = landmarks;
    bad.target_points[0][0] = f64::NAN;
    assert!(bad.align().is_err());
}

#[test]
fn optional_amass_fitter_runs_with_synthetic_supplied_source_and_explicit_map() {
    use anny_core::{
        inverter::FitOptions,
        model::ModelMetadata,
        smpl::{SmplConfig, SmplKind, SmplModel},
        ModelData,
    };
    let target = common::tiny();
    let n = 55;
    let mut d = ModelData {
        metadata: ModelMetadata {
            bone_labels: (0..n).map(|i| format!("j{i}")).collect(),
            bone_parents: std::iter::once(-1)
                .chain(std::iter::repeat_n(0, n - 1))
                .collect(),
            blendshape_labels: vec!["beta:0".into()],
        },
        ..Default::default()
    };
    for key in ["template_vertices", "faces", "base_mesh_vertex_indices"] {
        d.put(key, target.data.get(key).unwrap().clone());
    }
    d.put("blendshapes", Tensor::zeros(vec![1, 3, 3]));
    d.put("template_bone_heads", Tensor::zeros(vec![n, 3]));
    d.put(
        "template_bone_tails",
        Tensor::new(vec![n, 3], [0., 1., 0.].repeat(n)).unwrap(),
    );
    d.put("bone_heads_blendshapes", Tensor::zeros(vec![1, n, 3]));
    d.put("bone_tails_blendshapes", Tensor::zeros(vec![1, n, 3]));
    d.put(
        "bone_rolls_rotmat",
        Tensor::new(
            vec![1, n, 3, 3],
            [1., 0., 0., 0., 1., 0., 0., 0., 1.].repeat(n),
        )
        .unwrap(),
    );
    d.put(
        "vertex_bone_weights",
        Tensor::new(vec![3, 1], vec![1.; 3]).unwrap(),
    );
    d.put(
        "vertex_bone_indices",
        Tensor::indices(vec![3, 1], vec![0; 3]),
    );
    let source = SmplModel {
        data: d,
        config: SmplConfig {
            kind: SmplKind::Smplx,
            num_betas: 1,
            num_expression_coeffs: 0,
            pose_corrective: false,
            use_pca: false,
        },
        pose_mean: Tensor::zeros(vec![1, n, 3]),
        left_hand_components: None,
        right_hand_components: None,
    };
    source.validate().unwrap();
    let sequence = AmassSequence {
        betas: Tensor::zeros(vec![1]),
        root_orient: Tensor::zeros(vec![2, 3]),
        pose_body: Tensor::zeros(vec![2, 63]),
        pose_hand: Tensor::zeros(vec![2, 90]),
        pose_jaw: Tensor::zeros(vec![2, 3]),
        translations: Tensor::new(vec![2, 3], vec![0., 0., 0., 0.05, 0., 0.]).unwrap(),
        fps: Some(30.),
        gender: None,
    };
    let mut options = AmassFitOptions {
        pose_iterations: 1,
        shape: FitOptions {
            optimize_phenotypes: false,
            max_n_iters: Some(1),
            ..Default::default()
        },
        ..Default::default()
    };
    let result = fit_amass(&sequence, &source, &target, &VertexMap::Identity, &options).unwrap();
    assert_eq!(result.clip.frames.len(), 2);
    assert_eq!(result.source_frame_indices, vec![0, 1]);
    assert!(result.frame_errors.iter().flatten().all(|e| e.is_finite()));
    result.clip.validate(&target).unwrap();
    let map = VertexMap::Weighted {
        indices: vec![vec![0], vec![1], vec![2]],
        weights: vec![vec![1.]; 3],
    };
    let vertices = Tensor::new(vec![1, 3, 3], vec![0., 0., 0., 1., 0., 0., 0., 0., 1.]).unwrap();
    assert_eq!(map.map(&vertices, 3).unwrap().data, vertices.data);
    assert!(VertexMap::Identity.map(&vertices, 4).is_err());
    // Caller fitting/refinement settings must reach the per-frame pose stage
    // while the phenotype stage stays frozen.
    options.shape.post_gd = true;
    options.shape.post_gd_steps = 2;
    options.shape.post_gd_lr = 1e-2;
    options.shape.post_gd_optimize_local_changes = true;
    options.shape.post_gd_optimize_facial_actions = true;
    options
        .shape
        .multistart
        .insert("height".into(), vec![0.4, 0.6]);
    let refined = fit_amass(&sequence, &source, &target, &VertexMap::Identity, &options).unwrap();
    assert_eq!(refined.frame_refinement_losses.len(), 2);
    assert_eq!(
        refined.shape_refinement_losses.len(),
        3,
        "the retained shape stage honors caller refinement too"
    );
    for losses in &refined.frame_refinement_losses {
        assert_eq!(losses.len(), 3, "initial loss plus two Adam steps");
        assert!(losses.iter().all(|loss| loss.is_finite()));
    }
    options.max_frames = 1;
    assert!(fit_amass(&sequence, &source, &target, &VertexMap::Identity, &options).is_err());
}
