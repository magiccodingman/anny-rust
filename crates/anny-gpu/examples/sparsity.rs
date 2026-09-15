//! How sparse are the coefficients the evaluator really produces, and how much
//! of a real evaluation is the blendshape stage?
//!
//! `cargo run -p anny-gpu --release --example sparsity`

use anny_core::typed::{apply_blendshapes, TensorF32};
use anny_core::{Anny, ModelData, Parameters};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "output/ci-model.safetensors".to_string());
    let bytes = std::fs::read(&path)?;
    let model = ModelData::load(&path)?;
    let template = TensorF32::from_reference(model.get("template_vertices")?)?;
    let blendshapes = TensorF32::from_reference(model.get("blendshapes")?)?;
    let c = model.blendshape_count();

    let anny = Anny::from_bytes(&bytes, None)?;
    let params = Parameters::default();
    let session = anny.pose_session(&params)?;
    let coeffs = session.coefficients();
    let nonzero = coeffs.data.iter().filter(|v| **v != 0.0).count();
    println!(
        "coefficients: {} total, {} nonzero ({:.2}%)",
        coeffs.data.len(),
        nonzero,
        100.0 * nonzero as f64 / coeffs.data.len() as f64
    );

    let real = TensorF32::new(vec![1, c], coeffs.data.iter().map(|v| *v as f32).collect())?;
    let dense = TensorF32::new(
        vec![1, c],
        (0..c)
            .map(|k| ((k * 17) % 101) as f32 / 101.0 - 0.5)
            .collect(),
    )?;

    let time = |label: &str, f: &dyn Fn() -> Result<(), anny_core::Error>| {
        let start = Instant::now();
        for _ in 0..20 {
            f().expect("stage runs");
        }
        println!(
            "{label}: {:.3} ms",
            start.elapsed().as_secs_f64() * 1000.0 / 20.0
        );
    };

    time("apply_blendshapes, real coefficients (B=1) ", &|| {
        apply_blendshapes(&template, &blendshapes, &real).map(|_| ())
    });
    time("apply_blendshapes, dense synthetic (B=1) ", &|| {
        apply_blendshapes(&template, &blendshapes, &dense).map(|_| ())
    });
    time("Anny::rest_model (whole rest evaluation)  ", &|| {
        anny.rest_model(coeffs).map(|_| ())
    });
    time("Anny::pose_session (coeffs + rest model)  ", &|| {
        anny.pose_session(&params).map(|_| ())
    });
    Ok(())
}
