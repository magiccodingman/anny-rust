use anny_core::{
    config::*,
    distribution::*,
    math::*,
    model::{self, ModelMetadata},
    tensor::{Archive, Kind},
    *,
};
use serde_json::json;
fn tiny() -> Anny {
    let mut d = ModelData {
        metadata: ModelMetadata {
            bone_labels: vec!["root".into(), "joint".into()],
            bone_parents: vec![-1, 0],
            blendshape_labels: vec![
                "universal:fixture".into(),
                "facial_action:jawOpen".into(),
                "local_change:test-pos".into(),
                "local_change:test-neg".into(),
            ],
        },
        ..Default::default()
    };
    d.put(
        "template_vertices",
        Tensor::new(vec![3, 3], vec![0., 0., 0., 1., 0., 0., 0., 0., 1.]).unwrap(),
    );
    d.put("faces", Tensor::indices(vec![1, 3], vec![0, 1, 2]));
    d.put(
        "base_mesh_vertex_indices",
        Tensor::indices(vec![3], vec![0, 1, 2]),
    );
    let mut shapes = Tensor::zeros(vec![4, 3, 3]);
    shapes.data[9 + 7] = 0.2;
    shapes.data[18 + 3] = 0.5;
    shapes.data[27 + 3] = -0.3;
    d.put("blendshapes", shapes);
    d.put(
        "stacked_phenotype_blend_shapes_mask",
        Tensor::zeros(vec![1, 26]),
    );
    d.put(
        "template_bone_heads",
        Tensor::new(vec![2, 3], vec![0., 0., 0., 0., 0., 1.]).unwrap(),
    );
    d.put(
        "template_bone_tails",
        Tensor::new(vec![2, 3], vec![0., 1., 0., 0., 1., 1.]).unwrap(),
    );
    d.put("bone_heads_blendshapes", Tensor::zeros(vec![4, 2, 3]));
    d.put("bone_tails_blendshapes", Tensor::zeros(vec![4, 2, 3]));
    d.put(
        "bone_rolls_rotmat",
        Tensor::new(
            vec![1, 2, 3, 3],
            [1., 0., 0., 0., -1., 0., 0., 0., -1.].repeat(2),
        )
        .unwrap(),
    );
    d.put(
        "vertex_bone_weights",
        Tensor::new(vec![3, 2], vec![1., 0., 1., 0., 0., 1.]).unwrap(),
    );
    d.put(
        "vertex_bone_indices",
        Tensor::indices(vec![3, 2], vec![0, 1, 0, 1, 0, 1]),
    );
    Anny::from_model_data(
        d,
        AnnyConfig {
            rig: RigSpec::Name("makehuman".into()),
            facial_actions: Selection::Preset("all".into()),
            local_changes: Selection::Preset("all".into()),
            ..Default::default()
        },
    )
    .unwrap()
}
fn close(a: &[f64], b: &[f64], epsilon: f64) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert!((a - b).abs() <= epsilon, "{a} != {b}");
    }
}
#[test]
fn config_and_legacy_defaults() {
    let c = AnnyConfig::default();
    assert_eq!(c.pose_parameterization, PoseParameterization::LocalRef);
    assert_eq!(c.phenotype_labels().len(), 6);
    assert!(RigConfig::parse("anny-notoes-hand.L").is_ok());
    assert!(RigConfig::parse("anny-whoops").is_err());
    assert!(TopologyConfig::parse("anny-quads-full").is_ok());
    assert!(TopologyConfig::parse("default").is_err());
}
#[test]
fn interpolation_boundaries() {
    close(
        &linear_interpolation(0.25, &[0., 0.5, 1.], false).unwrap(),
        &[0.5, 0.5, 0.],
        0.,
    );
    close(
        &linear_interpolation(2., &[0., 1.], false).unwrap(),
        &[0., 1.],
        0.,
    );
    close(
        &linear_interpolation(2., &[0., 1.], true).unwrap(),
        &[-1., 2.],
        0.,
    );
    assert!(linear_interpolation(0., &[1., 0.], false).is_err());
}
#[test]
fn tensor_roundtrip_and_malformed_input() {
    let a = Archive {
        tensors: std::collections::BTreeMap::from([
            (
                "a".into(),
                Tensor::new(vec![2, 2], vec![1., 2., 3., 4.]).unwrap(),
            ),
            ("i".into(), Tensor::indices(vec![2], vec![4, 7])),
        ]),
        metadata: Default::default(),
    };
    let b = Archive::from_bytes(&a.to_bytes().unwrap()).unwrap();
    assert_eq!(b.tensors["a"].data, a.tensors["a"].data);
    assert_eq!(b.tensors["i"].kind, Kind::Index);
    assert!(Tensor::from_nested(&json!([[1, 2], [3]])).is_err());
    assert!(Tensor::new(vec![usize::MAX, 2], vec![]).is_err());
    assert!(Archive::from_bytes(b"not a tensor").is_err());
}
#[test]
fn config_model_cache_roundtrip() {
    let m = tiny();
    let bytes = m.data.archive(Some(&m.config)).unwrap().to_bytes().unwrap();
    let copy = Anny::from_bytes(&bytes, None).unwrap();
    assert_eq!(copy.data.metadata.bone_labels, m.data.metadata.bone_labels);
    close(
        &copy
            .forward(&Parameters::default())
            .unwrap()
            .get("vertices")
            .unwrap()
            .data,
        &m.forward(&Parameters::default())
            .unwrap()
            .get("vertices")
            .unwrap()
            .data,
        0.,
    );
}
#[test]
fn invalid_model_is_rejected() {
    let mut m = tiny();
    m.data.metadata.bone_parents = vec![1, 0];
    assert!(m.data.validate().is_err());
    let mut m = tiny();
    m.data.arrays.get_mut("faces").unwrap().data[0] = 100.;
    assert!(m.data.validate().is_err());
}
#[test]
fn signed_local_changes_and_facial_actions() {
    let m = tiny();
    let p = Parameters {
        local_changes_kwargs: json!({"test-pos":[1.,-1.]}),
        facial_actions: json!({"jawOpen":0.5}),
        ..Default::default()
    };
    let out = m.forward(&p).unwrap();
    let v = out.get("vertices").unwrap();
    assert_eq!(v.shape, [2, 3, 3]);
    assert!((v.data[3] - 1.5).abs() < 1e-12);
    assert!((v.data[12] - 0.7).abs() < 1e-12);
    assert!((v.data[7] - 0.1).abs() < 1e-12);
}
#[test]
fn invalid_inputs_and_batching() {
    let m = tiny();
    assert!(m
        .forward(&Parameters {
            phenotype_kwargs: json!({"unknown":1}),
            ..Default::default()
        })
        .is_err());
    assert!(m
        .forward(&Parameters {
            phenotype_kwargs: json!({"height":[0.1,0.2],"weight":[0.2,0.4,0.5]}),
            ..Default::default()
        })
        .is_err());
    assert!(m
        .forward(&Parameters {
            pose_parameters: json!({"missing": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}),
            ..Default::default()
        })
        .is_err());
}
#[test]
fn pose_roundtrip_all_five_conventions() {
    let m = tiny();
    let mut p = Parameters::default();
    let mut pose = model::identity_poses(1, 2);
    write4(
        &rigid(
            &rotvec(&Vec3::new(0.1, 0.2, -0.3)),
            &Vec3::new(0.3, -0.2, 0.1),
        ),
        &mut pose.data[..16],
    );
    write4(
        &rigid(&rotvec(&Vec3::new(0.2, 0.1, 0.3)), &Vec3::zeros()),
        &mut pose.data[16..],
    );
    p.pose_parameters = pose.nested_json();
    let original = m.forward(&p).unwrap();
    for mode in [
        PoseParameterization::LocalRef,
        PoseParameterization::LocalBone,
        PoseParameterization::LocalBoneWorld,
        PoseParameterization::World,
        PoseParameterization::WorldOrient,
    ] {
        let q = m.pose_parameters(&original, mode).unwrap();
        let output = m
            .forward(&Parameters {
                pose_parameters: q.nested_json(),
                pose_parameterization: Some(mode),
                ..Default::default()
            })
            .unwrap();
        close(
            &output.get("vertices").unwrap().data,
            &original.get("vertices").unwrap().data,
            1e-10,
        );
    }
}
#[test]
fn basic_kinematics_and_dqs() {
    assert!(propagation_order(&[-1, 0, 0, 1]).is_ok());
    assert!(propagation_order(&[1, 0]).is_err());
    let r = rotvec(&Vec3::new(0., 0., 0.3));
    let t = rigid(&r, &Vec3::new(1., 2., 3.));
    let v = Vec3::new(0.1, 0.2, 0.3);
    let result = dqs_point(&v, std::iter::once((1., dual_quaternion(&t)))).unwrap();
    close(result.as_slice(), point(&t, &v).as_slice(), 1e-12);
}
#[test]
fn rigid_registration_roundtrip() {
    let a = [
        Vec3::new(0., 0., 0.),
        Vec3::new(1., 0., 0.),
        Vec3::new(0., 1., 0.),
        Vec3::new(0., 0., 1.),
    ];
    let t = rigid(&rotvec(&Vec3::new(0.2, -0.3, 0.4)), &Vec3::new(1., 2., 3.));
    let b = a.map(|p| point(&t, &p));
    let estimate = rigid_registration(&a, &b, &[1.; 4], true).unwrap();
    close(estimate.as_slice(), t.as_slice(), 1e-12);
}
#[test]
fn mesh_helpers() {
    let m = tiny();
    assert_eq!(
        tools::boundary_edges(m.data.get("faces").unwrap())
            .unwrap()
            .len(),
        3
    );
    assert!(tools::triangle_intersects_sat(
        [Vec3::zeros(), Vec3::x(), Vec3::y()],
        [
            Vec3::new(0.2, 0.2, -1.),
            Vec3::new(0.2, 0.2, 1.),
            Vec3::new(0.5, 0.2, 0.)
        ]
    ));
}
#[test]
fn keypoint_dense_and_sparse() {
    let v = Tensor::new(vec![1, 3, 3], vec![0., 0., 0., 1., 0., 0., 0., 1., 0.]).unwrap();
    let dense = tools::KeypointsRegressor {
        labels: vec!["p".into()],
        weights: Tensor::new(vec![1, 3], vec![0.25, 0.75, 0.]).unwrap(),
        indices: None,
    };
    let sparse = tools::KeypointsRegressor {
        labels: vec!["p".into()],
        weights: Tensor::new(vec![1, 2], vec![0.25, 0.75]).unwrap(),
        indices: Some(Tensor::indices(vec![1, 2], vec![0, 1])),
    };
    close(
        &dense.regress(&v).unwrap().data,
        &sparse.regress(&v).unwrap().data,
        0.,
    );
}
#[test]
fn calibrated_distributions() {
    let mapping = MorphologicalAgeMapping {
        anny_age_anchors: vec![0., 0.5, 1.],
        morphological_age_anchors: vec![0., 30., 90.],
    };
    let age = mapping.morphological_to_anny_age(60.).unwrap();
    assert_eq!(age, 0.75);
    assert_eq!(mapping.anny_to_morphological_age(age).unwrap(), 60.);
    let beta = ConditionalBetaDistribution {
        age_anchors: vec![0., 1.],
        alpha_anchors: vec![2., 2.],
        beta_anchors: vec![2., 2.],
    };
    assert!((beta.log_probability(0.5, 0.5).unwrap() - 1.5f64.ln()).abs() < 1e-12);
    let mut a = ShapeRng::new(42);
    let mut b = ShapeRng::new(42);
    for _ in 0..20 {
        assert_eq!(
            beta.sample(0.2, &mut a).unwrap(),
            beta.sample(0.2, &mut b).unwrap()
        );
    }
}
#[test]
fn inversion_rigid_smoke() {
    let m = tiny();
    let pose = rigid(
        &rotvec(&Vec3::new(0.1, 0.2, 0.3)),
        &Vec3::new(0.3, 0.4, 0.5),
    );
    let mut tp = model::identity_poses(1, 2);
    write4(&pose, &mut tp.data[..16]);
    let target = m
        .forward(&Parameters {
            pose_parameters: tp.nested_json(),
            pose_parameterization: Some(PoseParameterization::LocalBone),
            ..Default::default()
        })
        .unwrap();
    let fitter = inverter::AnnyInverter::new(&m, Default::default()).unwrap();
    let fit = fitter
        .fit(
            target.get("vertices").unwrap(),
            &inverter::FitOptions {
                optimize_phenotypes: false,
                max_n_iters: Some(0),
                ..Default::default()
            },
        )
        .unwrap();
    assert!(fit.mean_vertex_error[0] < 1e-8);
}
