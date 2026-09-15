//! Measures the GPU blendshape contraction against the CPU reference.
//!
//! `cargo run -p anny-gpu --release --example parity_blendshapes -- <model> [adapter-substring]`
//!
//! Prints, per batch size: the max absolute deviation from `anny_core`'s own
//! f32 kernel, how many elements are bit-identical, and the wall time of both
//! paths so the crossover (if any) is measured rather than assumed.

use anny_core::typed::{apply_blendshapes, TensorF32};
use anny_core::ModelData;
use anny_gpu::blendshapes::{BlendshapeKernel, Weights};
use anny_gpu::Gpu;

fn coefficients(batch: usize, c: usize) -> TensorF32 {
    // Deterministic, spread over [-0.5, 0.5], with every 7th coefficient left
    // at zero so the CPU's zero-skipping path is exercised too.
    let mut data = Vec::with_capacity(batch * c);
    for b in 0..batch {
        for k in 0..c {
            let r = ((b * 7919 + k * 104729) % 1000) as f32 / 1000.0 - 0.5;
            data.push(if k % 7 == 0 { 0.0 } else { r });
        }
    }
    TensorF32::new(vec![batch, c], data).expect("coefficient shape")
}

fn deviation(gpu_out: &[f32], reference: &[f32]) -> (f64, f64, usize) {
    let mut max_abs = 0f64;
    let mut max_rel = 0f64;
    let mut identical = 0usize;
    for (g, r) in gpu_out.iter().zip(reference) {
        let g = *g as f64;
        let r = *r as f64;
        if g.to_bits() == r.to_bits() {
            identical += 1;
        }
        max_abs = max_abs.max((g - r).abs());
        max_rel = max_rel.max((g - r).abs() / r.abs().max(1e-30));
    }
    (max_abs, max_rel, identical)
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values[values.len() / 2]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "output/ci-model.safetensors".into());
    let needle = std::env::args().nth(2);
    let data = ModelData::load(&model_path)?;
    let gpu = Gpu::open_matching(needle.as_deref())?;
    println!("model: {model_path}");
    println!("adapter: {} [{:?}]", gpu.adapter_name, gpu.backend);

    let template = TensorF32::from_reference(data.get("template_vertices")?)?;
    let blendshapes = TensorF32::from_reference(data.get("blendshapes")?)?;
    let c = data.blendshape_count();
    let size = template.data.len();
    println!(
        "vertices: {} | blendshapes: {c} | flatten size: {size}",
        data.vertex_count()
    );

    let kernel = BlendshapeKernel::new(&gpu)?;
    println!(
        "\n{:>5} | {:>12} | {:>10} | {:>9} | {:>10} | {:>10} | {:>7}",
        "batch", "max abs diff", "max rel", "identical", "cpu ms", "gpu ms", "speedup"
    );
    for batch in [1usize, 4, 16, 64, 256] {
        let coeffs = coefficients(batch, c);
        let reference = apply_blendshapes(&template, &blendshapes, &coeffs)?;
        let gpu_out = kernel.run(
            &gpu,
            &template.data,
            &blendshapes.data,
            &coeffs.data,
            batch,
            c,
        )?;
        assert_eq!(gpu_out.len(), reference.data.len(), "output length");

        let mut max_abs = 0f64;
        let mut max_rel = 0f64;
        let mut identical = 0usize;
        for (g, r) in gpu_out.iter().zip(&reference.data) {
            let g = *g as f64;
            let r = *r as f64;
            if g.to_bits() == r.to_bits() {
                identical += 1;
            }
            max_abs = max_abs.max((g - r).abs());
            let denom = r.abs().max(1e-30);
            max_rel = max_rel.max((g - r).abs() / denom);
        }

        let cpu_ms = median(
            (0..5)
                .map(|_| {
                    let t = std::time::Instant::now();
                    let _ = apply_blendshapes(&template, &blendshapes, &coeffs).unwrap();
                    t.elapsed().as_secs_f64() * 1e3
                })
                .collect(),
        );
        let gpu_ms = median(
            (0..5)
                .map(|_| {
                    let t = std::time::Instant::now();
                    let _ = kernel
                        .run(
                            &gpu,
                            &template.data,
                            &blendshapes.data,
                            &coeffs.data,
                            batch,
                            c,
                        )
                        .unwrap();
                    t.elapsed().as_secs_f64() * 1e3
                })
                .collect(),
        );
        println!(
            "{batch:>5} | {max_abs:>12.3e} | {max_rel:>10.3e} | {identical:>9} | {cpu_ms:>10.3} | {gpu_ms:>10.3} | {:>6.2}x",
            cpu_ms / gpu_ms
        );
    }

    // Is the GPU inside the precision class the project already ships? Measure
    // the f32 kernel against the f64 reference for the same coefficients.
    let coeffs = coefficients(1, c);
    let f32_out = apply_blendshapes(&template, &blendshapes, &coeffs)?;
    let mut coeffs64 = anny_core::Tensor::zeros(vec![1, c]);
    for (d, s) in coeffs64.data.iter_mut().zip(&coeffs.data) {
        *d = *s as f64;
    }
    let f64_out = anny_core::model::apply_blendshapes(
        data.get("template_vertices")?,
        data.get("blendshapes")?,
        &coeffs64,
    )?;
    let mut f32_vs_f64 = 0f64;
    for (a, b) in f32_out.data.iter().zip(&f64_out.data) {
        f32_vs_f64 = f32_vs_f64.max((*a as f64 - b).abs());
    }
    println!("\nf32 kernel vs f64 reference (batch 1): max abs diff {f32_vs_f64:.3e}");

    // Interactive shape: the weights stay on the device, only coefficients travel.
    let weights = Weights::upload(&gpu, &template.data, &blendshapes.data, c);
    println!(
        "\nresident weights: {:.1} MB uploaded once",
        (template.data.len() + blendshapes.data.len()) as f64 * 4.0 / 1e6
    );
    println!(
        "{:>5} | {:>12} | {:>9} | {:>10} | {:>10} | {:>7}",
        "batch", "max abs diff", "identical", "cpu ms", "gpu ms", "speedup"
    );
    for batch in [1usize, 4, 16, 64, 256] {
        let coeffs = coefficients(batch, c);
        let reference = apply_blendshapes(&template, &blendshapes, &coeffs)?;
        let gpu_out = weights.run(&gpu, &kernel, &coeffs.data, batch)?;
        let (max_abs, _, identical) = deviation(&gpu_out, &reference.data);
        let cpu_ms = median(
            (0..5)
                .map(|_| {
                    let t = std::time::Instant::now();
                    let _ = apply_blendshapes(&template, &blendshapes, &coeffs).unwrap();
                    t.elapsed().as_secs_f64() * 1e3
                })
                .collect(),
        );
        let gpu_ms = median(
            (0..5)
                .map(|_| {
                    let t = std::time::Instant::now();
                    let _ = weights.run(&gpu, &kernel, &coeffs.data, batch).unwrap();
                    t.elapsed().as_secs_f64() * 1e3
                })
                .collect(),
        );
        println!(
            "{batch:>5} | {max_abs:>12.3e} | {identical:>9} | {cpu_ms:>10.3} | {gpu_ms:>10.3} | {:>6.2}x",
            cpu_ms / gpu_ms
        );
    }
    Ok(())
}
