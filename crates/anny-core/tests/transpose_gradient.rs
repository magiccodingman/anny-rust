mod common;
use anny_core::{
    differentiation::{jvp, vjp, ParameterDirection, ParameterSelection},
    Parameters, Tensor,
};
use serde_json::json;
use std::collections::BTreeMap;

#[test]
fn analytic_transpose_identity_and_gradient_selection() {
    let m = common::tiny();
    let p = Parameters {
        local_changes_kwargs: json!({"test-pos":0.2}),
        ..Default::default()
    };
    let s = ParameterSelection {
        local_changes: m.local_change_labels.clone(),
        facial_actions: m.facial_action_labels.clone(),
        bone_rotations: m.data.metadata.bone_labels.clone(),
        bone_translations: vec!["root".into()],
        ..Default::default()
    };
    let mut d = ParameterDirection::default();
    d.local_changes.insert("test-pos".into(), 0.3);
    d.facial_actions.insert("jawOpen".into(), -0.2);
    d.bone_rotations.insert("root".into(), [0.2, -0.1, 0.3]);
    d.bone_rotations.insert("joint".into(), [-0.1, 0.2, 0.05]);
    d.bone_translations.insert("root".into(), [0.3, 0.2, -0.1]);
    let cot = BTreeMap::from([(
        "vertices".into(),
        Tensor::new(
            vec![1, 3, 3],
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        )
        .unwrap(),
    )]);
    let jt = vjp(&m, &p, &s, &cot).unwrap();
    let j = jvp(&m, &p, &d).unwrap();
    let lhs: f64 = j
        .get("vertices")
        .unwrap()
        .data
        .iter()
        .zip(&cot["vertices"].data)
        .map(|(a, b)| a * b)
        .sum();
    let mut rhs = d.local_changes["test-pos"] * jt.local_changes["test-pos"]
        + d.facial_actions["jawOpen"] * jt.facial_actions["jawOpen"];
    for (a, b) in [
        (&d.bone_rotations, &jt.bone_rotations),
        (&d.bone_translations, &jt.bone_translations),
    ] {
        for (name, v) in a {
            rhs += v.iter().zip(b[name]).map(|(x, y)| x * y).sum::<f64>();
        }
    }
    assert!((lhs - rhs).abs() < 1e-12);
    assert!(jt.phenotypes.is_empty());
    let h = 1e-5;
    let objective = |x: f64| {
        let p = Parameters {
            local_changes_kwargs: json!({"test-pos":x}),
            ..p.clone()
        };
        m.forward(&p)
            .unwrap()
            .get("vertices")
            .unwrap()
            .data
            .iter()
            .zip(&cot["vertices"].data)
            .map(|(a, b)| a * b)
            .sum::<f64>()
    };
    assert!(
        (jt.local_changes["test-pos"] - (objective(0.2 + h) - objective(0.2 - h)) / (2. * h)).abs()
            < 1e-9
    );
}
#[test]
fn cotangents_are_validated_even_without_selected_controls() {
    let m = common::tiny();
    let mut s = ParameterSelection::default();
    let p = Parameters::default();
    assert!(vjp(&m, &p, &s, &BTreeMap::new()).is_err());
    let mut cot = BTreeMap::from([("vertices".into(), Tensor::zeros(vec![1, 2, 3]))]);
    assert!(vjp(&m, &p, &s, &cot).is_err());
    cot.insert("vertices".into(), Tensor::zeros(vec![1, 3, 3]));
    assert!(vjp(&m, &p, &s, &cot).unwrap().phenotypes.is_empty());
    s.local_changes = vec!["test-pos".into(), "test-pos".into()];
    assert!(vjp(&m, &p, &s, &cot).is_err());
    s.local_changes = vec!["missing".into()];
    assert!(vjp(&m, &p, &s, &cot).is_err());
}
