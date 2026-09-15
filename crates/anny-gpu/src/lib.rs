//! GPU evaluation kernels for Anny, running on wgpu (Vulkan / WebGPU).
//!
//! The CPU evaluator in `anny_core` is the reference. Every kernel here is
//! verified against it by `crates/anny-gpu/tests/parity.rs`, which reports the
//! measured deviation rather than asserting a tolerance it did not measure.

use thiserror::Error;

pub mod blendshapes;

#[derive(Debug, Error)]
pub enum GpuError {
    #[error("no GPU adapter available: {0}")]
    NoAdapter(String),
    #[error("wgpu error: {0}")]
    Wgpu(#[from] wgpu::Error),
    #[error("device request failed: {0}")]
    Device(#[from] wgpu::RequestDeviceError),
    #[error("buffer mapping failed: {0}")]
    Map(#[from] wgpu::BufferAsyncError),
    #[error("device poll failed: {0}")]
    Poll(#[from] wgpu::PollError),
}

/// An open device plus the adapter it came from.
pub struct Gpu {
    pub adapter_name: String,
    pub backend: wgpu::Backend,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl Gpu {
    /// Open the first adapter `wgpu` offers.
    pub fn open() -> Result<Self, GpuError> {
        Self::open_matching(None)
    }

    /// Open an adapter whose name contains `needle`, if given. Used to pin the
    /// software adapters (llvmpipe) so parity runs against a second
    /// implementation, not only the same driver twice.
    pub fn open_matching(needle: Option<&str>) -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::from_env().unwrap_or(wgpu::Backends::all()),
            ..Default::default()
        });
        let mut names = Vec::new();
        for adapter in instance.enumerate_adapters(wgpu::Backends::all()) {
            let info = adapter.get_info();
            let matches = needle
                .is_none_or(|n| info.name.contains(n) || format!("{:?}", info.backend).contains(n));
            names.push(format!("{} [{:?}]", info.name, info.backend));
            if !matches {
                continue;
            }
            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))?;
            return Ok(Self {
                adapter_name: info.name,
                backend: info.backend,
                device,
                queue,
            });
        }
        Err(GpuError::NoAdapter(match needle {
            Some(n) => format!("none matching {n:?}; saw {names:?}"),
            None => format!("saw {names:?}"),
        }))
    }
}
