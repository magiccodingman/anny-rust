//! The blendshape contraction on the GPU.
//!
//! Mirrors `apply_blendshapes` in `anny-core/src/kernels/evaluation.rs`:
//! `out[b][i] = template[i] + sum_k coeff[b][k] * blendshapes[k][i]`.
//! The CPU skips zero coefficients; the shader skips them too, so both touch
//! only the active rows. Real coefficient sets are sparse — the default
//! character's 624 coefficients have 32 nonzero (5.13%) — and the parity check
//! still includes zero coefficients on purpose, since skipping must not change
//! the result.

use crate::{Gpu, GpuError};

pub const BLENDSHAPES_WGSL: &str = r#"
struct Params {
    size: u32,
    blendshapes: u32,
    batch: u32,
    pad: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> template_values: array<f32>;
@group(0) @binding(2) var<storage, read> blendshape_values: array<f32>;
@group(0) @binding(3) var<storage, read> coefficients: array<f32>;
@group(0) @binding(4) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size || gid.y >= params.batch) {
        return;
    }
    var acc: f32 = template_values[gid.x];
    for (var k: u32 = 0u; k < params.blendshapes; k = k + 1u) {
        let w = coefficients[gid.y * params.blendshapes + k];
        if (w == 0.0) {
            continue;
        }
        acc = acc + w * blendshape_values[k * params.size + gid.x];
    }
    result[gid.y * params.size + gid.x] = acc;
}
"#;

/// Matches `@workgroup_size` above.
pub const WORKGROUP: u32 = 64;

pub struct BlendshapeKernel {
    pipeline: wgpu::ComputePipeline,
    params: wgpu::Buffer,
}

/// Model weights kept resident on the device.
///
/// The one-shot path re-uploads the blendshape tensor (102 MB for the CI
/// model) on every call, which costs more than the kernel saves at batch 1.
/// A real interactive character uploads the weights once and then only sends
/// coefficients per pose.
pub struct Weights {
    template_buf: wgpu::Buffer,
    blendshapes_buf: wgpu::Buffer,
    size: usize,
    c: usize,
    blendshape_len: usize,
}

fn as_bytes(v: &[f32]) -> &[u8] {
    // f32 arrays are copied to the GPU verbatim; this host is little-endian,
    // which is what WGSL storage buffers expect.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Hand the bytes of a mapped readback buffer back to the caller.
///
/// Native waits on the device until wgpu pumps the mapping callback. The web
/// backend has no blocking poll — the browser resolves the map from its own task
/// queue — so there the mapping is awaited through a promise. Both are `async`
/// so the call sites read the same on every target.
#[cfg(not(target_arch = "wasm32"))]
async fn readback(gpu: &Gpu, buffer: &wgpu::Buffer) -> Result<Vec<f32>, GpuError> {
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    gpu.device.poll(wgpu::PollType::Wait)?;
    rx.recv().expect("mapping callback dropped")?;
    let data = slice.get_mapped_range();
    let out = to_f32(&data);
    drop(data);
    buffer.unmap();
    Ok(out)
}

#[cfg(target_arch = "wasm32")]
async fn readback(gpu: &Gpu, buffer: &wgpu::Buffer) -> Result<Vec<f32>, GpuError> {
    use wasm_bindgen::JsValue;
    // The device is only needed for the native poll; the browser drives itself.
    let _ = gpu;
    let slice = buffer.slice(..);
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        let resolve = resolve.clone();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = resolve.call1(&JsValue::UNDEFINED, &JsValue::from(r.is_ok()));
        });
    });
    wasm_bindgen_futures::JsFuture::from(promise)
        .await
        .map_err(|_| GpuError::WebReadback)?;
    let data = slice.get_mapped_range();
    let out = to_f32(&data);
    drop(data);
    buffer.unmap();
    Ok(out)
}

impl Weights {
    /// Uploads the template and the full blendshape tensor once.
    pub fn upload(gpu: &Gpu, template: &[f32], blendshapes: &[f32], c: usize) -> Self {
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let mk = |label: &str, bytes: &[u8]| {
            let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.len() as u64,
                usage: storage,
                mapped_at_creation: false,
            });
            gpu.queue.write_buffer(&buf, 0, bytes);
            buf
        };
        Self {
            template_buf: mk("template (resident)", as_bytes(template)),
            blendshapes_buf: mk("blendshapes (resident)", as_bytes(blendshapes)),
            size: template.len(),
            c,
            blendshape_len: blendshapes.len(),
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// Poses the resident weights. Only the coefficients travel per call.
    ///
    /// Blocking wrapper; native only. The web backend has to await its readback,
    /// so browsers go through [`run_async`](Self::run_async).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn run(
        &self,
        gpu: &Gpu,
        kernel: &BlendshapeKernel,
        coefficients: &[f32],
        batch: usize,
    ) -> Result<Vec<f32>, GpuError> {
        pollster::block_on(self.run_async(gpu, kernel, coefficients, batch))
    }

    /// Poses the resident weights on whichever backend is available.
    pub async fn run_async(
        &self,
        gpu: &Gpu,
        kernel: &BlendshapeKernel,
        coefficients: &[f32],
        batch: usize,
    ) -> Result<Vec<f32>, GpuError> {
        if batch == 0 || self.c == 0 || self.size == 0 {
            return Err(GpuError::InvalidInput(
                "batch, coefficient count and output size must be nonzero".into(),
            ));
        }
        let expected_coefficients = batch
            .checked_mul(self.c)
            .ok_or_else(|| GpuError::InvalidInput("coefficient shape overflow".into()))?;
        if coefficients.len() != expected_coefficients {
            return Err(GpuError::InvalidInput(format!(
                "expected {expected_coefficients} coefficients for batch {batch} x {}, got {}",
                self.c,
                coefficients.len()
            )));
        }
        let expected_blendshapes = self
            .c
            .checked_mul(self.size)
            .ok_or_else(|| GpuError::InvalidInput("blendshape shape overflow".into()))?;
        if self.blendshape_len != expected_blendshapes {
            return Err(GpuError::InvalidInput(format!(
                "resident blendshape buffer has {} values; expected {expected_blendshapes}",
                self.blendshape_len
            )));
        }
        let out_len = batch
            .checked_mul(self.size)
            .ok_or_else(|| GpuError::InvalidInput("output shape overflow".into()))?;
        let coeffs_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("coefficients"),
            size: (coefficients.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&coeffs_buf, 0, as_bytes(coefficients));
        let out_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("result"),
            size: (out_len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (out_len * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut param_bytes = [0u8; 16];
        for (i, v) in [self.size as u32, self.c as u32, batch as u32, 0u32]
            .iter()
            .enumerate()
        {
            param_bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        gpu.queue.write_buffer(kernel.params_ref(), 0, &param_bytes);

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("anny blendshapes"),
            layout: &kernel.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: kernel.params_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.template_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.blendshapes_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: coeffs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&kernel.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((self.size as u32).div_ceil(WORKGROUP), batch as u32, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &readback, 0, (out_len * 4) as u64);
        gpu.queue.submit(Some(encoder.finish()));

        self::readback(gpu, &readback).await
    }
}

impl BlendshapeKernel {
    /// Access for the resident path, which builds its own bind group.
    fn params_ref(&self) -> &wgpu::Buffer {
        &self.params
    }

    pub fn new(gpu: &Gpu) -> Result<Self, GpuError> {
        let module = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("anny blendshapes"),
                source: wgpu::ShaderSource::Wgsl(BLENDSHAPES_WGSL.into()),
            });
        let pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("anny blendshapes"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let params = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("anny blendshape params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Self { pipeline, params })
    }

    /// `template`/`blendshapes` are the flattened CPU tensors: template is
    /// `size` long, blendshapes is `c * size`, coefficients is `batch * c`.
    /// Returns `batch * size` values.
    ///
    /// Blocking wrapper; native only. Browsers use
    /// [`run_async`](Self::run_async).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn run(
        &self,
        gpu: &Gpu,
        template: &[f32],
        blendshapes: &[f32],
        coefficients: &[f32],
        batch: usize,
        c: usize,
    ) -> Result<Vec<f32>, GpuError> {
        pollster::block_on(self.run_async(gpu, template, blendshapes, coefficients, batch, c))
    }

    /// The same contraction, awaiting the readback instead of blocking on it.
    pub async fn run_async(
        &self,
        gpu: &Gpu,
        template: &[f32],
        blendshapes: &[f32],
        coefficients: &[f32],
        batch: usize,
        c: usize,
    ) -> Result<Vec<f32>, GpuError> {
        let size = template.len();
        if batch == 0 || c == 0 || size == 0 {
            return Err(GpuError::InvalidInput(
                "batch, coefficient count and template size must be nonzero".into(),
            ));
        }
        let expected_blendshapes = c
            .checked_mul(size)
            .ok_or_else(|| GpuError::InvalidInput("blendshape shape overflow".into()))?;
        if blendshapes.len() != expected_blendshapes {
            return Err(GpuError::InvalidInput(format!(
                "expected {expected_blendshapes} blendshape values for {c} x {size}, got {}",
                blendshapes.len()
            )));
        }
        let expected_coefficients = batch
            .checked_mul(c)
            .ok_or_else(|| GpuError::InvalidInput("coefficient shape overflow".into()))?;
        if coefficients.len() != expected_coefficients {
            return Err(GpuError::InvalidInput(format!(
                "expected {expected_coefficients} coefficients for batch {batch} x {c}, got {}",
                coefficients.len()
            )));
        }
        let out_len = batch
            .checked_mul(size)
            .ok_or_else(|| GpuError::InvalidInput("output shape overflow".into()))?;
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let mk = |label: &str, bytes: &[u8], usage| {
            let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.len() as u64,
                usage,
                mapped_at_creation: false,
            });
            gpu.queue.write_buffer(&buf, 0, bytes);
            buf
        };
        let template_buf = mk("template", as_bytes(template), storage);
        let blendshapes_buf = mk("blendshapes", as_bytes(blendshapes), storage);
        let coeffs_buf = mk("coefficients", as_bytes(coefficients), storage);
        let out_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("result"),
            size: (out_len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params = [size as u32, c as u32, batch as u32, 0u32];
        let mut param_bytes = [0u8; 16];
        for (i, v) in params.iter().enumerate() {
            param_bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        gpu.queue.write_buffer(&self.params, 0, &param_bytes);

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("anny blendshapes"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: template_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: blendshapes_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: coeffs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("anny blendshapes"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("anny blendshapes"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((size as u32).div_ceil(WORKGROUP), batch as u32, 1);
        }
        gpu.queue.submit(Some(encoder.finish()));

        let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (out_len * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("anny readback"),
            });
        encoder.copy_buffer_to_buffer(&out_buf, 0, &readback, 0, (out_len * 4) as u64);
        gpu.queue.submit(Some(encoder.finish()));

        self::readback(gpu, &readback).await
    }
}
