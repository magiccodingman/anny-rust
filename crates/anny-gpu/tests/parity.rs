//! GPU kernels must reproduce the shipped f32 evaluator to within the precision
//! class the project already has, and the resident-weight path must reproduce
//! the one-shot path exactly.
//!
//! Both tests skip when no model or no GPU adapter is present, so `cargo test`
//! stays green on machines without either.

use anny_core::typed::{apply_blendshapes, TensorF32};
use anny_core::ModelData;
use anny_gpu::blendshapes::{BlendshapeKernel, Weights};
use anny_gpu::Gpu;

/// Absolute bound: three orders below the 1e-3 tolerance the Unity bake tests use.
const MAX_ABS_DIFF: f64 = 1e-5;

fn model_path() -> Option<String> {
    if let Ok(p) = std::env::var("ANNY_MODEL") {
        return std::path::Path::new(&p).exists().then_some(p);
    }
    // Tests run with the crate directory as cwd; the model lives at the workspace root.
    let candidate = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("output/ci-model.safetensors");
    candidate.exists().then(|| candidate.display().to_string())
}

fn deviation(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x as f64 - *y as f64).abs())
        .fold(0f64, f64::max)
}

fn coefficients(batch: usize, c: usize) -> TensorF32 {
    let mut data = Vec::with_capacity(batch * c);
    for b in 0..batch {
        for k in 0..c {
            data.push((((b * 31 + k * 17) % 101) as f32 / 101.0) - 0.5);
        }
    }
    TensorF32::new(vec![batch, c], data).expect("coefficient shape")
}

/// Loads the CI model and an adapter, or explains why the test is skipped.
fn fixture() -> Option<(Gpu, ModelData, TensorF32, TensorF32, usize)> {
    let path = match model_path() {
        Some(p) => p,
        None => {
            println!("SKIP: no model (set ANNY_MODEL or create output/ci-model.safetensors)");
            return None;
        }
    };
    let gpu = match Gpu::open_matching(None) {
        Ok(g) => g,
        Err(e) => {
            println!("SKIP: no usable GPU adapter: {e}");
            return None;
        }
    };
    let data = ModelData::load(&path).expect("model loads");
    let c = data.blendshape_count();
    let template = TensorF32::from_reference(data.get("template_vertices").unwrap()).unwrap();
    let blendshapes = TensorF32::from_reference(data.get("blendshapes").unwrap()).unwrap();
    Some((gpu, data, template, blendshapes, c))
}

#[test]
fn gpu_blendshapes_agree_with_the_f32_evaluator() {
    let Some((gpu, _data, template, blendshapes, c)) = fixture() else {
        return;
    };
    let kernel = BlendshapeKernel::new(&gpu).expect("kernel builds");
    for batch in [1usize, 4, 16, 64] {
        let coeffs = coefficients(batch, c);
        let reference = apply_blendshapes(&template, &blendshapes, &coeffs).unwrap();
        let gpu_out = kernel
            .run(
                &gpu,
                &template.data,
                &blendshapes.data,
                &coeffs.data,
                batch,
                c,
            )
            .expect("kernel runs");
        let diff = deviation(&gpu_out, &reference.data);
        println!("batch {batch:>3}: max abs diff {diff:.3e}");
        assert!(
            diff <= MAX_ABS_DIFF,
            "batch {batch}: GPU differs from the f32 evaluator by {diff:.3e}"
        );
    }
}

#[test]
fn resident_weights_reproduce_the_one_shot_path_exactly() {
    let Some((gpu, _data, template, blendshapes, c)) = fixture() else {
        return;
    };
    let kernel = BlendshapeKernel::new(&gpu).expect("kernel builds");
    let weights = Weights::upload(&gpu, &template.data, &blendshapes.data, c);
    let coeffs = coefficients(4, c);
    let one_shot = kernel
        .run(&gpu, &template.data, &blendshapes.data, &coeffs.data, 4, c)
        .expect("kernel runs");
    let resident = weights
        .run(&gpu, &kernel, &coeffs.data, 4)
        .expect("resident run");
    assert_eq!(
        one_shot, resident,
        "resident weights must reproduce the one-shot path bit for bit"
    );
}
