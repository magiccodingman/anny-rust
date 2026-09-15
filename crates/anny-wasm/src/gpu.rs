//! Browser GPU path: the wgpu blendshape contraction over WebGPU, next to the
//! f32 CPU reference that `crates/anny-gpu/tests/parity.rs` checks it against.
//!
//! Coefficients are built here rather than in JavaScript, so a browser run
//! exercises the same sparse workload the native measurements use.
use crate::js_error;
use anny_core::typed::{apply_blendshapes, TensorF32};
use anny_core::{Anny, Parameters, Tensor};
use anny_gpu::blendshapes::BlendshapeKernel;
use anny_gpu::Gpu;
use std::sync::OnceLock;
use wasm_bindgen::prelude::*;

/// The adapter the last GPU call opened, for the harness to report.
static ADAPTER: OnceLock<String> = OnceLock::new();

fn reference(model: &Anny, name: &str) -> Result<TensorF32, JsValue> {
    let tensor: &Tensor = model.data.get(name).map_err(js_error)?;
    TensorF32::from_reference(tensor).map_err(js_error)
}

fn parts(bytes: &[u8]) -> Result<(TensorF32, TensorF32, usize, usize), JsValue> {
    let model = Anny::from_bytes(bytes, None).map_err(js_error)?;
    let template = reference(&model, "template_vertices")?;
    let blendshapes = reference(&model, "blendshapes")?;
    let c = model.data.blendshape_count();
    let size = template.data.len();
    Ok((template, blendshapes, c, size))
}

/// The default character's own coefficients — 32 of 624 rows active — tiled
/// `batch` times with those rows rotated and rescaled slightly per row, so a
/// batch carries real sparsity rather than a dense synthetic fill.
#[wasm_bindgen]
pub fn default_coefficients(bytes: &[u8], batch: usize) -> Result<js_sys::Float32Array, JsValue> {
    let model = Anny::from_bytes(bytes, None).map_err(js_error)?;
    let session = model
        .pose_session(&Parameters::default())
        .map_err(js_error)?;
    let real = session.coefficients();
    let c = model.data.blendshape_count();
    let batch = batch.max(1);
    let mut out = vec![0.0f32; batch * c];
    for b in 0..batch {
        for (k, v) in real.data.iter().enumerate() {
            let v = *v as f32;
            if v == 0.0 {
                continue;
            }
            let dest = (k + b * 5) % c;
            out[b * c + dest] = v * (1.0 + 0.1 * ((b + k) % 3) as f32);
        }
    }
    Ok(js_sys::Float32Array::from(&out[..]))
}

/// The f32 CPU reference contraction, one coefficient row at a time.
#[wasm_bindgen]
pub fn cpu_blendshapes(
    bytes: &[u8],
    coefficients: Vec<f32>,
    batch: usize,
) -> Result<js_sys::Float32Array, JsValue> {
    let (template, blendshapes, c, size) = parts(bytes)?;
    let mut out = Vec::with_capacity(batch * size);
    for b in 0..batch {
        let row = TensorF32::new(vec![1, c], coefficients[b * c..(b + 1) * c].to_vec())
            .map_err(js_error)?;
        let posed = apply_blendshapes(&template, &blendshapes, &row).map_err(js_error)?;
        out.extend_from_slice(&posed.data);
    }
    Ok(js_sys::Float32Array::from(&out[..]))
}

/// The same contraction on the browser's GPU, through WebGPU.
#[wasm_bindgen]
pub async fn gpu_blendshapes(
    bytes: &[u8],
    coefficients: Vec<f32>,
    batch: usize,
) -> Result<js_sys::Float32Array, JsValue> {
    let (template, blendshapes, c, _) = parts(bytes)?;
    let gpu = Gpu::open_async(None).await.map_err(js_error)?;
    // Browsers redact the adapter name, so record the backend alongside it: that is
    // what shows the call really went through WebGPU.
    let _ = ADAPTER.set(format!("{} [{:?}]", gpu.adapter_name, gpu.backend));
    let kernel = BlendshapeKernel::new(&gpu).map_err(js_error)?;
    let out = kernel
        .run_async(
            &gpu,
            &template.data,
            &blendshapes.data,
            &coefficients,
            batch,
            c,
        )
        .await
        .map_err(js_error)?;
    Ok(js_sys::Float32Array::from(&out[..]))
}

/// Which adapter the GPU path opened, once it has run.
#[wasm_bindgen]
pub fn gpu_adapter_name() -> String {
    ADAPTER.get().cloned().unwrap_or_else(|| "none".to_string())
}
