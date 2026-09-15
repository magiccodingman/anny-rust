//! GPU kernels must reproduce the shipped f32 evaluator to within the precision
//! class the project already has, and the resident-weight path must reproduce
//! the one-shot path exactly.
//!
//! Both tests skip when no model or no GPU adapter is present, so `cargo test`
//! stays green on machines without either.

use anny_core::model::{Anny, Parameters};
use anny_core::typed::{apply_blendshapes, TensorF32};
use anny_core::{AnnyConfig, ModelData};
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

/// A skip is only allowed when an operator has opted into it. With
/// `ANNY_REQUIRE_GPU=1` set, a missing model or adapter fails the test instead,
/// so a run that never touched the GPU cannot be mistaken for a passing one.
fn skip(reason: &str) {
    if std::env::var("ANNY_REQUIRE_GPU")
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false)
    {
        panic!("ANNY_REQUIRE_GPU=1 but {reason}");
    }
    println!("SKIP: {reason}");
}

/// Loads the CI model and an adapter, or explains why the test is skipped.
fn fixture() -> Option<(Gpu, ModelData, TensorF32, TensorF32, usize)> {
    let path = match model_path() {
        Some(p) => p,
        None => {
            skip("no model (set ANNY_MODEL or create output/ci-model.safetensors)");
            return None;
        }
    };
    let gpu = match Gpu::open_matching(None) {
        Ok(g) => g,
        Err(e) => {
            skip(&format!("no usable GPU adapter: {e}"));
            return None;
        }
    };
    println!("adapter: {}", gpu.adapter_name);
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

/// Four phenotype vectors close to the default — each row nudges every phenotype a little, which is
/// how a batch is built in practice.
fn near_default(anny: &Anny) -> Parameters {
    let mut parameters = Parameters::default();
    let mut kwargs = serde_json::Map::new();
    for (index, label) in anny.phenotype_labels.iter().enumerate() {
        let values: Vec<f64> = (0..4)
            .map(|row| 0.5 + 0.02 * (((row + index) % 3) as f64 - 1.0))
            .collect();
        kwargs.insert(label.clone(), serde_json::json!(values));
    }
    parameters.phenotype_kwargs = serde_json::Value::Object(kwargs);
    parameters
}

/// Four phenotype vectors spread across the phenotype range: most blend shapes end up partly active,
/// which is the dense end of what the phenotype path can produce.
fn spread(anny: &Anny) -> Parameters {
    let mut parameters = Parameters::default();
    let mut kwargs = serde_json::Map::new();
    for (index, label) in anny.phenotype_labels.iter().enumerate() {
        let values: Vec<f64> = (0..4)
            .map(|row| 0.25 + 0.5 * (((row + index) % 4) as f64) / 3.0)
            .collect();
        kwargs.insert(label.clone(), serde_json::json!(values));
    }
    parameters.phenotype_kwargs = serde_json::Value::Object(kwargs);
    parameters
}

/// The comparisons above run on synthetic coefficient vectors. Production coefficients arrive from
/// the phenotype path instead and are what the kernel would actually be handed, so the same bound is
/// asserted on those.
///
/// Sparsity is a property of the phenotype values rather than of the path, so it is asserted only
/// where it is the published figure (the default character) and printed for the rest; the active
/// counts below are the evidence that the inputs are real ones.
#[test]
fn production_coefficients_agree_with_the_f32_evaluator() {
    let Some((gpu, data, template, blendshapes, c)) = fixture() else {
        return;
    };
    let anny = Anny::from_model_data(data, AnnyConfig::default()).expect("model builds");
    let kernel = BlendshapeKernel::new(&gpu).expect("kernel builds");

    for (label, parameters, batch, sparse) in [
        ("default phenotype", Parameters::default(), 1usize, true),
        (
            "four near-default phenotypes",
            near_default(&anny),
            4,
            false,
        ),
        ("four spread phenotypes", spread(&anny), 4, false),
    ] {
        let coefficients = anny.coefficients(&parameters).expect("coefficients");
        let coefficients = TensorF32::from_reference(&coefficients).expect("f32 coefficients");
        let active = coefficients.data.iter().filter(|v| **v != 0.0).count();
        // The kernel's zero-skip works on exact zeros, so the crossover turns on `active`; the
        // second count says how much of that is a real shape rather than an epsilon.
        let significant = coefficients.data.iter().filter(|v| v.abs() > 1e-3).count();
        println!(
            "{label}: {active} of {c} coefficients active, {significant} above 1e-3, batch {batch}"
        );

        assert_eq!(
            coefficients.shape,
            vec![batch, c],
            "{label}: coefficient shape"
        );
        if sparse {
            assert!(
                active < c / 4,
                "{label}: the default character's coefficients are sparse, but {active} of {c} are active"
            );
        }

        let reference = apply_blendshapes(&template, &blendshapes, &coefficients).unwrap();
        let gpu_out = kernel
            .run(
                &gpu,
                &template.data,
                &blendshapes.data,
                &coefficients.data,
                batch,
                c,
            )
            .expect("kernel runs");
        let diff = deviation(&gpu_out, &reference.data);
        println!("{label}: max abs diff {diff:.3e}");
        assert!(
            diff <= MAX_ABS_DIFF,
            "{label}: GPU differs from the f32 evaluator by {diff:.3e}"
        );
    }
}
