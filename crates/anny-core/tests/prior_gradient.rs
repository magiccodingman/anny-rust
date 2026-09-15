use anny_core::{distribution::*, prior::shape_prior_gradient};
use std::collections::BTreeMap;
fn distribution() -> SimpleShapeDistribution {
    let b = ConditionalBetaDistribution {
        age_anchors: vec![0., 0.4, 1.],
        alpha_anchors: vec![1.2, 2.3, 3.8],
        beta_anchors: vec![2.2, 1.9, 3.],
    };
    let g = ConditionalBetaDistribution {
        age_anchors: vec![0., 0.6, 1.],
        alpha_anchors: vec![2.1, 3.3, 2.8],
        beta_anchors: vec![1.2, 4., 2.],
    };
    SimpleShapeDistribution {
        age_mapping: MorphologicalAgeMapping {
            anny_age_anchors: vec![0., 1.],
            morphological_age_anchors: vec![0., 100.],
        },
        boys: ["height", "weight", "muscle", "proportions"]
            .map(|s| (s.into(), b.clone()))
            .into(),
        girls: ["height", "weight", "muscle", "proportions"]
            .map(|s| (s.into(), g.clone()))
            .into(),
        phenotype_labels: vec![],
    }
}
#[test]
fn mixture_prior_gradient_matches_central_differences() {
    let d = distribution();
    for (age, gender) in [(0.27, 0.31), (0.77, 0.8), (-0.1, 0.65), (1.1, 0.65)] {
        let ph: BTreeMap<_, _> = [
            ("age", age),
            ("gender", gender),
            ("height", 0.3),
            ("weight", 0.65),
            ("muscle", 0.2),
            ("proportions", 0.73),
        ]
        .map(|(k, v)| (k.into(), v))
        .into();
        let (loss, grads) = shape_prior_gradient(&d, &ph).unwrap();
        assert!((loss - d.prior_loss(&ph).unwrap()).abs() < 1e-12);
        for (name, gradient) in grads {
            let mut p = ph.clone();
            let mut m = ph.clone();
            *p.get_mut(&name).unwrap() += 1e-5;
            *m.get_mut(&name).unwrap() -= 1e-5;
            let numeric = (d.prior_loss(&p).unwrap() - d.prior_loss(&m).unwrap()) / 2e-5;
            assert!(
                (gradient - numeric).abs() < 2e-6,
                "{name}: {gradient} != {numeric}"
            );
        }
    }
}
#[test]
fn clamped_and_invalid_inputs_are_explicit() {
    let d = distribution();
    let mut p: BTreeMap<_, _> = [("height".into(), -0.2), ("gender".into(), 1.2)].into();
    let (_, g) = shape_prior_gradient(&d, &p).unwrap();
    assert_eq!(g["height"], 0.);
    assert_eq!(g["gender"], 0.);
    p.insert("age".into(), f64::NAN);
    assert!(shape_prior_gradient(&d, &p).is_err());
}
