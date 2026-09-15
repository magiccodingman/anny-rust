//! Prints every adapter wgpu can see, then opens a device on each backend's
//! first adapter. Run: `cargo run -p anny-gpu --example adapters`.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapters = instance.enumerate_adapters(wgpu::Backends::all());
    println!("adapters: {}", adapters.len());
    for a in &adapters {
        let i = a.get_info();
        println!(
            "  {:?} | {} | type={:?} | driver={} {}",
            i.backend, i.name, i.device_type, i.driver, i.driver_info
        );
    }
    // Open a device on the discrete adapter, mirroring how the kernels are used.
    let gpu = anny_gpu::Gpu::open()?;
    println!("\nopened: {} on {:?}", gpu.adapter_name, gpu.backend);
    let f = gpu.device.features();
    println!(
        "feature checks: timestamp_query={} push_constants={} f16={} subgroups={}",
        f.contains(wgpu::Features::TIMESTAMP_QUERY),
        f.contains(wgpu::Features::PUSH_CONSTANTS),
        f.contains(wgpu::Features::SHADER_F16),
        f.contains(wgpu::Features::SUBGROUP)
    );
    println!(
        "max workgroup: x={} y={} z={}",
        gpu.device.limits().max_compute_workgroup_size_x,
        gpu.device.limits().max_compute_workgroup_size_y,
        gpu.device.limits().max_compute_workgroup_size_z
    );
    println!(
        "max storage buffer binding: {} bytes",
        gpu.device.limits().max_storage_buffer_binding_size
    );
    Ok(())
}
