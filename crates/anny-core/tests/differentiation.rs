mod common;
use anny_core::{
    config::*,
    differentiation::{jvp, ParameterDirection},
    math::*,
    model, Anny, Parameters, Tensor,
};
use serde_json::json;
fn perturb(model: &Anny, p: &Parameters, d: &ParameterDirection, h: f64) -> Parameters {
    let mut out = p.clone();
    for (field, labels, direction, default) in [
        (
            &mut out.phenotype_kwargs,
            &model.phenotype_labels,
            &d.phenotypes,
            0.5,
        ),
        (
            &mut out.local_changes_kwargs,
            &model.local_change_labels,
            &d.local_changes,
            0.,
        ),
        (
            &mut out.facial_actions,
            &model.facial_action_labels,
            &d.facial_actions,
            0.,
        ),
    ] {
        let data = model::parse_values(field, labels, default, "test").unwrap();
        let map: serde_json::Map<_, _> = labels
            .iter()
            .enumerate()
            .map(|(i, n)| {
                (
                    n.clone(),
                    json!(data.data[i] + h * direction.get(n).unwrap_or(&0.)),
                )
            })
            .collect();
        *field = map.into();
    }
    let mut pose = model::parse_pose(&p.pose_parameters, &model.data.metadata.bone_labels).unwrap();
    for (i, row) in pose.data.chunks_exact_mut(16).enumerate() {
        let current = mat4(row);
        let name = &model.data.metadata.bone_labels[i];
        let omega = d.bone_rotations.get(name).copied().unwrap_or([0.; 3]);
        let translation_delta = d.bone_translations.get(name).copied().unwrap_or([0.; 3]);
        let r = rotvec(&(vec3(&omega) * h)) * rotation(&current);
        let t = translation(&current) + vec3(&translation_delta) * h;
        write4(&rigid(&r, &t), row);
    }
    out.pose_parameters = pose.nested_json();
    out
}
fn check(model: &Anny, p: &Parameters, d: &ParameterDirection, tolerance: f64) -> f64 {
    let h = 1e-5;
    let gradient = jvp(model, p, d).unwrap();
    let plus = model.forward(&perturb(model, p, d, h)).unwrap();
    let minus = model.forward(&perturb(model, p, d, -h)).unwrap();
    let mut largest: f64 = 0.;
    for (name, values) in &gradient.arrays {
        let p = &plus.get(name).unwrap().data;
        let m = &minus.get(name).unwrap().data;
        for ((&g, &a), &b) in values.data.iter().zip(p).zip(m) {
            let finite = (a - b) / (2. * h);
            let e = (g - finite).abs();
            largest = largest.max(e);
            assert!(
                e < tolerance,
                "{name}: analytic {g} vs finite {finite}, difference {e}"
            );
        }
    }
    largest
}
#[test]
fn analytic_directions_match_finite_differences_for_pose_shape_and_skinning() {
    let mut d = ParameterDirection::default();
    d.local_changes.insert("test-pos".into(), 0.4);
    d.facial_actions.insert("jawOpen".into(), 0.2);
    d.bone_rotations.insert("root".into(), [0.1, -0.2, 0.3]);
    d.bone_translations.insert("root".into(), [0.2, 0.1, -0.1]);
    d.bone_rotations.insert("joint".into(), [-0.2, 0.15, 0.1]);
    for cached in [false, true] {
        for skin in [SkinningMethod::Lbs, SkinningMethod::Dqs] {
            let mut m = common::tiny();
            m.config.skinning_method = skin;
            m.data
                .arrays
                .get_mut("bone_heads_blendshapes")
                .unwrap()
                .data[2 * 6 + 3] = 0.1;
            m.data
                .arrays
                .get_mut("bone_tails_blendshapes")
                .unwrap()
                .data[2 * 6 + 3] = 0.15;
            if cached {
                let mut rig = m.config.rig.resolve().unwrap();
                rig.bone_orientation = BoneOrientation::Cached;
                m.config.rig = RigSpec::Config(rig);
                m.data.put(
                    "bone_template_orientation_matrices",
                    Tensor::new(
                        vec![2, 3, 3],
                        [2., 0., 0., 0., 3., 0., 0., 0., 4.].repeat(2),
                    )
                    .unwrap(),
                );
                let mut cov = Tensor::zeros(vec![4, 2, 3, 3]);
                cov.data[2 * 18 + 1] = 0.15;
                cov.data[1 * 18 + 9 + 3] = -0.25;
                m.data.put("bone_orientation_blendshapes", cov);
            }
            let m = Anny::from_model_data(m.data, m.config).unwrap();
            for mode in [
                PoseParameterization::LocalBone,
                PoseParameterization::LocalRef,
                PoseParameterization::LocalBoneWorld,
                PoseParameterization::World,
                PoseParameterization::WorldOrient,
            ] {
                let p = Parameters {
                    pose_parameterization: Some(mode),
                    local_changes_kwargs: json!({"test-pos":0.3}),
                    facial_actions: json!({"jawOpen":0.2}),
                    ..Default::default()
                };
                println!("cached {cached} skin {skin:?} mode {mode:?}");
                check(&m, &p, &d, 2e-5);
            }
        }
    }
}
#[test]
fn differentiation_rejects_unsupported_or_invalid_inputs() {
    let m = common::tiny();
    let mut d = ParameterDirection::default();
    d.phenotypes.insert("imaginary".into(), 1.);
    assert!(jvp(&m, &Parameters::default(), &d).is_err());
    d = Default::default();
    d.bone_rotations.insert("root".into(), [f64::NAN, 0., 0.]);
    assert!(jvp(&m, &Parameters::default(), &d).is_err());
    let p = Parameters {
        phenotype_kwargs: json!({"height":[0.4,0.6]}),
        ..Default::default()
    };
    assert!(jvp(&m, &p, &ParameterDirection::default()).is_err());
    let zero = jvp(&m, &Parameters::default(), &ParameterDirection::default()).unwrap();
    assert!(zero.arrays.values().flat_map(|t| &t.data).all(|&x| x == 0.));
}
#[test]
#[ignore = "real asset first-order derivative qualification"]
fn real_body_directions() {
    let assets = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
    for (rig, skin) in [
        ("anny", SkinningMethod::Lbs),
        ("anny", SkinningMethod::Dqs),
        ("makehuman", SkinningMethod::Lbs),
        ("makehuman-procrustes", SkinningMethod::Lbs),
        ("soma", SkinningMethod::Lbs),
    ] {
        let config = AnnyConfig {
            rig: RigSpec::Name(rig.into()),
            skinning_method: skin,
            phenotypes: "all".into(),
            local_changes: Selection::all(),
            facial_actions: Selection::all(),
            ..Default::default()
        };
        let model = anny_core::assets::AssetStore::new(&assets)
            .build(&config)
            .unwrap();
        let mut p = Parameters {
            phenotype_kwargs: json!({"height":0.61,"weight":0.43,"age":0.52}),
            ..Default::default()
        };
        let mut direction = ParameterDirection::default();
        direction.phenotypes.insert("height".into(), 0.3);
        direction.phenotypes.insert("weight".into(), -0.2);
        direction.phenotypes.insert("african".into(), 0.1);
        let local = &model.local_change_labels[0];
        p.local_changes_kwargs = json!({local:0.17});
        direction.local_changes.insert(local.clone(), 0.05);
        let face = &model.facial_action_labels[0];
        p.facial_actions = json!({face:0.11});
        direction.facial_actions.insert(face.clone(), -0.08);
        let root = model.data.metadata.bone_labels[0].clone();
        direction
            .bone_rotations
            .insert(root.clone(), [0.03, -0.02, 0.01]);
        direction
            .bone_translations
            .insert(root, [0.01, 0.02, -0.03]);
        let largest = check(&model, &p, &direction, 5e-4);
        println!("{rig} {skin:?}: maximum JVP error {largest:.9e}");
    }
}
