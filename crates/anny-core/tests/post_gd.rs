mod common;
use anny_core::{
    inverter::{AnnyInverter, FitOptions, InverterOptions},
    operations, Parameters,
};
use serde_json::json;

#[test]
fn inverter_post_gd_is_opt_in_and_reports_actual_adam_steps() {
    let m = common::tiny();
    let target = m
        .forward(&Parameters {
            local_changes_kwargs: json!({"test-pos":0.5}),
            facial_actions: json!({"jawOpen":0.4}),
            ..Default::default()
        })
        .unwrap();
    let fit = AnnyInverter::new(&m, InverterOptions::default()).unwrap();
    let opts = FitOptions {
        optimize_phenotypes: false,
        max_n_iters: Some(0),
        post_gd_steps: 200,
        post_gd_lr: 0.01,
        post_gd_optimize_local_changes: true,
        post_gd_optimize_facial_actions: true,
        ..Default::default()
    };
    let baseline = fit.fit(target.get("vertices").unwrap(), &opts).unwrap();
    assert!(baseline.post_gd_losses.is_empty());
    let enabled = FitOptions {
        post_gd: true,
        ..opts
    };
    let result = fit.fit(target.get("vertices").unwrap(), &enabled).unwrap();
    assert_eq!(result.post_gd_losses.len(), 201);
    assert!(
        result.mean_vertex_error[0] < baseline.mean_vertex_error[0] * 0.02,
        "{} -> {}",
        baseline.mean_vertex_error[0],
        result.mean_vertex_error[0]
    );
    assert!(result.post_gd_losses.last().unwrap() < &(result.post_gd_losses[0] * 0.001));
    let rerun = m.forward(&result.parameters).unwrap();
    assert!(rerun
        .get("vertices")
        .unwrap()
        .data
        .iter()
        .zip(&result.output.get("vertices").unwrap().data)
        .all(|(a, b)| (a - b).abs() < 1e-12));
    assert!(result.parameters.phenotype_kwargs.is_object());
    let wrong = FitOptions {
        post_gd_prior_weight: 0.1,
        ..enabled
    };
    assert!(fit.fit(target.get("vertices").unwrap(), &wrong).is_err());
}

#[test]
fn shared_query_surface_exposes_real_gradients_and_refinement() {
    let m = common::tiny();
    let target = m
        .forward(&Parameters {
            local_changes_kwargs: json!({"test-pos":0.6}),
            ..Default::default()
        })
        .unwrap();
    let request = json!({"operation":"refine","target":target.get("vertices").unwrap(),"options":{"steps":100,"learning_rate":0.03,"selection":{"local_changes":["test-pos"]}}});
    let output: serde_json::Value =
        serde_json::from_str(&operations::execute_json(&m, &request.to_string()).unwrap()).unwrap();
    assert_eq!(output["iterations"], 100);
    let losses = output["losses"].as_array().unwrap();
    assert!(losses.last().unwrap().as_f64().unwrap() < losses[0].as_f64().unwrap() * 0.001);
    let jvp = json!({"operation":"jvp","direction":{"local_changes":{"test-pos":1.0}}});
    assert!(operations::execute_json(&m, &jvp.to_string()).is_ok());
    let vjp = json!({"operation":"vjp","selection":{"local_changes":["test-pos"]},"cotangents":{"vertices":target.get("vertices").unwrap()}});
    let result: serde_json::Value =
        serde_json::from_str(&operations::execute_json(&m, &vjp.to_string()).unwrap()).unwrap();
    assert!(result["local_changes"]["test-pos"]
        .as_f64()
        .unwrap()
        .is_finite());
}
