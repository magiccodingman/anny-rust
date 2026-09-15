//! GPU vs CPU on coefficient sets shaped like real workloads.
//!
//! Real coefficient sets are sparse (the default character: 32 of 624 nonzero),
//! and both the CPU kernel and the shader skip zeros, so the crossover depends
//! on how many coefficients a workload actually switches on. This measures both
//! extremes with the same code path.
//!
//! `cargo run -p anny-gpu --release --example real_batch -- [adapter-substring]`

use anny_core::typed::{apply_blendshapes, TensorF32};
use anny_core::{Anny, ModelData, Parameters};
use anny_gpu::blendshapes::{BlendshapeKernel, Weights};
use anny_gpu::Gpu;
use std::time::Instant;

const BATCHES: [usize; 5] = [1, 4, 16, 64, 256];
const REPEATS: usize = 10;

/// `n` coefficient sets at the real sparsity: the session's active pattern,
/// shifted per pose so every pose differs.
fn sparse_batch(real: &[f32], n: usize) -> Vec<f32> {
    let c = real.len();
    let active: Vec<(usize, f32)> = real
        .iter()
        .enumerate()
        .filter(|(_, v)| **v != 0.0)
        .map(|(i, v)| (i, *v))
        .collect();
    let mut out = vec![0.0f32; n * c];
    for b in 0..n {
        for (i, v) in &active {
            out[b * c + (i * 7 + b * 13) % c] = *v;
        }
    }
    out
}

/// `n` coefficient sets with every coefficient switched on.
fn dense_batch(c: usize, n: usize) -> Vec<f32> {
    (0..n * c)
        .map(|i| ((i * 17) % 101) as f32 / 101.0 - 0.5)
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let needle = std::env::args().nth(1);
    let path = std::env::var("ANNY_MODEL").unwrap_or_else(|_| "output/ci-model.safetensors".into());
    let bytes = std::fs::read(&path)?;
    let model = ModelData::load(&path)?;
    let template = TensorF32::from_reference(model.get("template_vertices")?)?;
    let blendshapes = TensorF32::from_reference(model.get("blendshapes")?)?;
    let c = model.blendshape_count();
    let size = template.data.len();

    let anny = Anny::from_bytes(&bytes, None)?;
    let session = anny.pose_session(&Parameters::default())?;
    let real: Vec<f32> = session
        .coefficients()
        .data
        .iter()
        .map(|v| *v as f32)
        .collect();
    let active = real.iter().filter(|v| **v != 0.0).count();

    let gpu = Gpu::open_matching(needle.as_deref())?;
    let kernel = BlendshapeKernel::new(&gpu)?;
    let weights = Weights::upload(&gpu, &template.data, &blendshapes.data, c);
    println!("adapter: {}", gpu.adapter_name);
    println!(
        "size={size} blendshapes={c} active={active} ({:.2}%)",
        100.0 * active as f64 / c as f64
    );

    for (label, sparse) in [("sparse (real 5.13%)", true), ("dense (all 624 on)", false)] {
        println!("\n{label}");
        println!(
            "{:>6} {:>12} {:>12} {:>8} {:>12}",
            "batch", "cpu ms", "gpu ms", "speedup", "max abs"
        );
        for n in BATCHES {
            let coeffs = if sparse {
                sparse_batch(&real, n)
            } else {
                dense_batch(c, n)
            };
            let t = TensorF32::new(vec![n, c], coeffs.clone())?;
            let start = Instant::now();
            let mut cpu = Vec::new();
            for _ in 0..REPEATS {
                cpu = apply_blendshapes(&template, &blendshapes, &t)?.data;
            }
            let cpu_ms = start.elapsed().as_secs_f64() * 1000.0 / REPEATS as f64;

            let start = Instant::now();
            let mut gpu_out = Vec::new();
            for _ in 0..REPEATS {
                gpu_out = weights.run(&gpu, &kernel, &coeffs, n)?;
            }
            let gpu_ms = start.elapsed().as_secs_f64() * 1000.0 / REPEATS as f64;

            let worst = cpu
                .iter()
                .zip(gpu_out.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!(
                "{n:>6} {cpu_ms:>12.3} {gpu_ms:>12.3} {:>7.2}x {worst:>12.3e}",
                cpu_ms / gpu_ms
            );
        }
    }
    Ok(())
}
