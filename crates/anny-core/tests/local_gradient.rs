mod common;
use anny_core::{
    differentiation::{jvp, ParameterDirection},
    Parameters,
};
use serde_json::json;

#[test]
fn zero_local_uses_upstream_masked_branch_subgradient() {
    let model = common::tiny();
    let mut direction = ParameterDirection::default();
    direction.local_changes.insert("test-pos".into(), 0.25);
    // The fixture has +0.5 and -0.3 signed targets. Upstream intentionally
    // gives both branches a derivative at zero: 0.5 - (-0.3), not ReLU's zero
    // and not the central-difference average of the two one-sided slopes.
    for (value, slope) in [(0., 0.8), (0.2, 0.5), (-0.2, 0.3)] {
        let p = Parameters {
            local_changes_kwargs: json!({"test-pos":value}),
            ..Default::default()
        };
        let result = jvp(&model, &p, &direction).unwrap();
        let actual = &result.get("rest_vertices").unwrap().data;
        assert!((actual[3] - 0.25 * slope).abs() < 1e-12);
        assert!(actual.iter().enumerate().all(|(i, &v)| i == 3 || v == 0.));
    }
}
