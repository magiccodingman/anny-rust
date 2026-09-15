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
    #[error("adapter request failed: {0}")]
    RequestAdapter(#[from] wgpu::RequestAdapterError),
    #[error("wgpu error: {0}")]
    Wgpu(#[from] wgpu::Error),
    #[error("device request failed: {0}")]
    Device(#[from] wgpu::RequestDeviceError),
    #[error("buffer mapping failed: {0}")]
    Map(#[from] wgpu::BufferAsyncError),
    #[error("device poll failed: {0}")]
    Poll(#[from] wgpu::PollError),
    #[error("readback failed on the web backend")]
    WebReadback,
    #[error("invalid GPU input: {0}")]
    InvalidInput(String),
}

/// An open device plus the adapter it came from.
pub struct Gpu {
    pub adapter_name: String,
    pub backend: wgpu::Backend,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

/// Instances are created one at a time, process-wide.
///
/// `vkCreateInstance` `dlopen`s the Vulkan layers, and an interposing
/// `LD_PRELOAD` shim is not necessarily thread-safe: with NoMachine's
/// `libnxegl.so` loaded, three threads creating instances at once abort the
/// process with `double free or corruption (fasttop)` in ~25% of runs, and in
/// 0% of runs with `LD_PRELOAD` cleared. Creation happens once per process, so
/// serializing it costs nothing measurable.
#[cfg(not(target_arch = "wasm32"))]
static OPEN_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

impl Gpu {
    /// Open the first adapter `wgpu` offers (blocks; native only).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open() -> Result<Self, GpuError> {
        pollster::block_on(Self::open_async(None))
    }

    /// Open an adapter whose name contains `needle`, if given. Used to pin the
    /// software adapters (llvmpipe) so parity runs against a second
    /// implementation, not only the same driver twice.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open_matching(needle: Option<&str>) -> Result<Self, GpuError> {
        pollster::block_on(Self::open_async(needle))
    }

    /// Open an adapter and a device without blocking — the only way in on the
    /// web, where `navigator.gpu` hands out adapters asynchronously.
    ///
    /// WebGPU has no adapter enumeration, so `needle` only refines the choice
    /// natively; in a browser we get whichever adapter the user agent picks.
    pub async fn open_async(needle: Option<&str>) -> Result<Self, GpuError> {
        // `Instance::new` is the call that reaches `vkCreateInstance`, and with
        // it the loader's `dlopen` of the Vulkan layers, so that is the part
        // that must not run concurrently; the guard is released before any
        // await, so this does not make the returned future `!Send`.
        #[cfg(not(target_arch = "wasm32"))]
        let instance = {
            let _guard = OPEN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::from_env().unwrap_or(wgpu::Backends::all()),
                ..Default::default()
            })
        };
        #[cfg(target_arch = "wasm32")]
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..Default::default()
        });

        #[cfg(target_arch = "wasm32")]
        {
            let _ = needle;
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await?;
            let info = adapter.get_info();
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await?;
            Ok(Self {
                adapter_name: info.name,
                backend: info.backend,
                device,
                queue,
            })
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut names = Vec::new();
            for adapter in instance.enumerate_adapters(wgpu::Backends::all()) {
                let info = adapter.get_info();
                let matches = needle.is_none_or(|n| {
                    info.name.contains(n) || format!("{:?}", info.backend).contains(n)
                });
                names.push(format!("{} [{:?}]", info.name, info.backend));
                if !matches {
                    continue;
                }
                let (device, queue) = adapter
                    .request_device(&wgpu::DeviceDescriptor::default())
                    .await?;
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
}
