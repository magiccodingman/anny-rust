mod common;
use anny_core::{
    differentiation::ParameterSelection,
    math::{mat4, write4, Vec3},
    model::identity_poses,
    refinement::{refine, ReconstructionLoss, RefinementOptions},
    Parameters, PoseParameterization, Tensor,
};
use nalgebra::UnitQuaternion;
use serde_json::json;

#[test]
fn adam_updates_initially_zero_local_and_facial_controls() {
    let model = common::tiny();
    let target = model
        .forward(&Parameters {
            local_changes_kwargs: json!({"test-pos":0.6}),
            facial_actions: json!({"jawOpen":0.7}),
            ..Default::default()
        })
        .unwrap();
    let options = RefinementOptions {
        steps: 160,
        learning_rate: 0.03,
        selection: Some(ParameterSelection {
            local_changes: model.local_change_labels.clone(),
            facial_actions: model.facial_action_labels.clone(),
            ..Default::default()
        }),
        ..Default::default()
    };
    let result = refine(
        &model,
        target.get("vertices").unwrap(),
        &Parameters::default(),
        &options,
        None,
    )
    .unwrap();
    assert!(
        result.losses.last().unwrap() < &(result.losses[0] * 1e-5),
        "{:?}",
        result.losses
    );
    assert_eq!(result.losses.len(), options.steps + 1);
    assert_eq!(result.iterations, options.steps);
    let generated = model.forward(&result.parameters).unwrap();
    let mse: f64 = generated
        .get("vertices")
        .unwrap()
        .data
        .iter()
        .zip(&target.get("vertices").unwrap().data)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / 9.;
    assert!((mse - result.losses.last().unwrap()).abs() < 1e-12);
}

#[test]
fn absolute_rotation_vector_chain_rule_converges_from_nonzero_pose() {
    let model = common::tiny();
    let make = |w: Vec3, translation: [f64; 3]| {
        let mut pose = identity_poses(1, 2);
        let mut h = mat4(&pose.data[..16]);
        h.fixed_view_mut::<3, 3>(0, 0).copy_from(
            UnitQuaternion::from_scaled_axis(w)
                .to_rotation_matrix()
                .matrix(),
        );
        for a in 0..3 {
            h[(a, 3)] = translation[a];
        }
        write4(&h, &mut pose.data[..16]);
        Parameters {
            pose_parameters: pose.nested_json(),
            pose_parameterization: Some(PoseParameterization::LocalBone),
            ..Default::default()
        }
    };
    let initial = make(Vec3::new(0.2, -0.3, 0.1), [0.1, -0.1, 0.2]);
    let desired = make(Vec3::new(-0.1, 0.2, 0.4), [0.3, 0.1, -0.1]);
    let target = model.forward(&desired).unwrap();
    let options = RefinementOptions {
        steps: 200,
        learning_rate: 0.02,
        selection: Some(ParameterSelection {
            bone_rotations: vec!["root".into()],
            bone_translations: vec!["root".into()],
            ..Default::default()
        }),
        ..Default::default()
    };
    let result = refine(
        &model,
        target.get("vertices").unwrap(),
        &initial,
        &options,
        None,
    )
    .unwrap();
    assert!(
        result.losses.last().unwrap() < &(result.losses[0] * 1e-6),
        "{} -> {:?}",
        result.losses[0],
        result.losses.last()
    );
}

#[test]
fn batched_targets_are_independent_and_constraints_are_enforced() {
    let model = common::tiny();
    let desired = Parameters {
        local_changes_kwargs: json!({"test-pos":[2.0,-2.0]}),
        facial_actions: json!({"jawOpen":[2.0,-2.0]}),
        ..Default::default()
    };
    let target = model.forward(&desired).unwrap();
    let options = RefinementOptions {
        steps: 60,
        learning_rate: 0.1,
        selection: Some(ParameterSelection {
            local_changes: model.local_change_labels.clone(),
            facial_actions: model.facial_action_labels.clone(),
            ..Default::default()
        }),
        ..Default::default()
    };
    let result = refine(
        &model,
        target.get("vertices").unwrap(),
        &Parameters::default(),
        &options,
        None,
    )
    .unwrap();
    let lo = anny_core::model::parse_values(
        &result.parameters.local_changes_kwargs,
        &model.local_change_labels,
        0.,
        "locals",
    )
    .unwrap();
    let fa = anny_core::model::parse_values(
        &result.parameters.facial_actions,
        &model.facial_action_labels,
        0.,
        "face",
    )
    .unwrap();
    assert_eq!(lo.shape[0], 2);
    assert!(lo.data[0] > 0.99 && lo.data[1] < -0.99);
    assert!(lo.data.iter().all(|v| (-1.0..=1.0).contains(v)));
    assert!(fa.data.iter().all(|v| (0.0..=1.0).contains(v)));
}

#[test]
fn invalid_refinement_options_are_rejected_before_optimization() {
    let model = common::tiny();
    let p = Parameters::default();
    let out = model.forward(&p).unwrap();
    let target = out.get("vertices").unwrap();
    for options in [
        RefinementOptions {
            learning_rate: 0.,
            ..Default::default()
        },
        RefinementOptions {
            learning_rate: f64::NAN,
            ..Default::default()
        },
        RefinementOptions {
            prior_weight: 1.,
            ..Default::default()
        },
        RefinementOptions {
            loss: ReconstructionLoss::Huber { delta: 0. },
            ..Default::default()
        },
        RefinementOptions {
            selection: Some(ParameterSelection {
                bone_translations: vec!["joint".into()],
                ..Default::default()
            }),
            ..Default::default()
        },
        RefinementOptions {
            selection: Some(ParameterSelection {
                local_changes: vec!["bad".into()],
                ..Default::default()
            }),
            ..Default::default()
        },
    ] {
        assert!(refine(&model, target, &p, &options, None).is_err());
    }
    assert!(refine(
        &model,
        &Tensor::zeros(vec![1, 4, 3]),
        &p,
        &RefinementOptions::default(),
        None
    )
    .is_err());
}

#[test]
fn huber_objective_and_zero_step_behavior_are_explicit() {
    let model = common::tiny();
    let initial = Parameters::default();
    let mut target = model
        .forward(&initial)
        .unwrap()
        .get("vertices")
        .unwrap()
        .clone();
    target.data[0] += 10.;
    let options = RefinementOptions {
        steps: 0,
        selection: Some(ParameterSelection::default()),
        loss: ReconstructionLoss::Huber { delta: 0.1 },
        ..Default::default()
    };
    let result = refine(&model, &target, &initial, &options, None).unwrap();
    assert!((result.losses[0] - 0.1 * (10. - 0.05) / 9.).abs() < 1e-12);
    assert_eq!(result.losses.len(), 1);
    assert_eq!(result.iterations, 0);
}

#[test]
fn shared_initial_phenotypes_are_validated() {
    let model = common::tiny();
    let initial = Parameters {
        phenotype_kwargs: json!({"height":[0.2,0.8]}),
        ..Default::default()
    };
    let target = model.forward(&initial).unwrap();
    let options = RefinementOptions {
        steps: 0,
        shared_phenotypes: true,
        ..Default::default()
    };
    assert!(refine(
        &model,
        target.get("vertices").unwrap(),
        &initial,
        &options,
        None
    )
    .is_err());
}

#[test]
fn shared_phenotype_logits_use_the_sum_of_batch_gradients() {
    let mut model = common::tiny();
    model
        .data
        .arrays
        .get_mut("stacked_phenotype_blend_shapes_mask")
        .unwrap()
        .data[17] = 1.;
    model.data.arrays.get_mut("blendshapes").unwrap().data[3] = 0.4;
    let initial = Parameters {
        phenotype_kwargs: json!({"height":0.3}),
        facial_actions: json!({"jawOpen":[0.2,0.7]}),
        ..Default::default()
    };
    let desired = Parameters {
        phenotype_kwargs: json!({"height":0.8}),
        ..initial.clone()
    };
    let target = model.forward(&desired).unwrap();
    let options = RefinementOptions {
        steps: 160,
        learning_rate: 0.1,
        shared_phenotypes: true,
        selection: Some(ParameterSelection {
            phenotypes: vec!["height".into()],
            ..Default::default()
        }),
        ..Default::default()
    };
    let result = refine(
        &model,
        target.get("vertices").unwrap(),
        &initial,
        &options,
        None,
    )
    .unwrap();
    let ph = anny_core::model::parse_values(
        &result.parameters.phenotype_kwargs,
        &model.phenotype_labels,
        0.5,
        "ph",
    )
    .unwrap();
    let h = model
        .phenotype_labels
        .iter()
        .position(|n| n == "height")
        .unwrap();
    assert_eq!(ph.data[h], ph.data[ph.shape[1] + h]);
    assert!((ph.data[h] - 0.8).abs() < 0.001);
    assert!(result.losses.last().unwrap() < &(result.losses[0] * 1e-5));
}

#[test]
fn supplied_calibrated_prior_drives_logit_optimization() {
    use anny_core::distribution::{
        ConditionalBetaDistribution, MorphologicalAgeMapping, SimpleShapeDistribution,
    };
    let model = common::tiny();
    let beta = ConditionalBetaDistribution {
        age_anchors: vec![0., 1.],
        alpha_anchors: vec![2., 2.],
        beta_anchors: vec![2., 2.],
    };
    let groups = ["height", "weight", "muscle", "proportions"]
        .map(|s| (s.to_string(), beta.clone()))
        .into();
    let prior = SimpleShapeDistribution {
        age_mapping: MorphologicalAgeMapping {
            anny_age_anchors: vec![0., 1.],
            morphological_age_anchors: vec![0., 100.],
        },
        boys: groups,
        girls: ["height", "weight", "muscle", "proportions"]
            .map(|s| (s.to_string(), beta.clone()))
            .into(),
        phenotype_labels: model.phenotype_labels.clone(),
    };
    let initial = Parameters {
        phenotype_kwargs: json!({"height":0.2}),
        ..Default::default()
    };
    let target = model.forward(&initial).unwrap();
    let options = RefinementOptions {
        steps: 100,
        learning_rate: 0.05,
        prior_weight: 1.,
        selection: Some(ParameterSelection {
            phenotypes: vec!["height".into()],
            ..Default::default()
        }),
        ..Default::default()
    };
    let result = refine(
        &model,
        target.get("vertices").unwrap(),
        &initial,
        &options,
        Some(&prior),
    )
    .unwrap();
    let ph = anny_core::model::parse_values(
        &result.parameters.phenotype_kwargs,
        &model.phenotype_labels,
        0.5,
        "ph",
    )
    .unwrap();
    let h = model
        .phenotype_labels
        .iter()
        .position(|n| n == "height")
        .unwrap();
    assert!((ph.data[h] - 0.5).abs() < 0.005);
    assert!(result.losses.last().unwrap() < &result.losses[0]);
}

#[test]
#[ignore = "real-asset analytic refinement qualification"]
fn real_body_analytic_refinement_reduces_reconstruction_loss() {
    let store = anny_core::assets::AssetStore::new(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data"),
    );
    let m = store.build(&Default::default()).unwrap();
    let target = m
        .forward(&Parameters {
            phenotype_kwargs: json!({"height":0.7}),
            ..Default::default()
        })
        .unwrap();
    let options = RefinementOptions {
        steps: 3,
        learning_rate: 0.05,
        selection: Some(ParameterSelection {
            phenotypes: vec!["height".into()],
            ..Default::default()
        }),
        ..Default::default()
    };
    let result = refine(
        &m,
        target.get("vertices").unwrap(),
        &Parameters::default(),
        &options,
        None,
    )
    .unwrap();
    println!("real height refinement: {:?}", result.losses);
    assert!(result.losses.last().unwrap() < &(result.losses[0] * 0.9));
}
