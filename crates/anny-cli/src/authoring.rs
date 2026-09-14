//! Native authoring commands. No interpreter or external model downloads.
use super::*;
use anny_core::{cache::ModelCache, fitting::*, precompute::*, transforms::*};
use serde_json::json;
use std::time::Instant;
fn write_json(path: &str, value: &impl serde::Serialize) -> Result<()> {
    parent(Path::new(path))?;
    std::fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}
fn number(opts: &BTreeMap<String, String>, name: &str, default: usize) -> Result<usize> {
    let n = opts
        .get(name)
        .map(|s| {
            s.parse::<usize>()
                .map_err(|_| error(format!("invalid --{name}")))
        })
        .transpose()?
        .unwrap_or(default);
    if n > 10_000 {
        return Err(error(format!("--{name} cannot exceed 10000")));
    }
    Ok(n)
}
pub fn help() {
    println!("export-calibration/export-keypoints --assets data [--config config.json] --output portable.json");
    println!("\nNative authoring:\ntransform (--assets data | --model prepared.safetensors) --operations operations.json --output prepared.safetensors\nprecompute-rig --assets data --rig anny|soma [--options options.json] --output covariance.safetensors\ncache-orientations (--assets data | --model prepared.safetensors) [--params reference.json] [--options options.json] --output prepared.safetensors\nrecompute-weights --assets data --output cleaned-weights.json\nfit --target mesh.obj|ply|stl|glb|gltf --correspondence index|closest-surface [--mesh-fit-options options.json] --output parameters.json\nquery (--assets data | --model prepared.safetensors) --request operation.json [--output response.json]\nbenchmark (--assets data | --model prepared.safetensors) [--params params.json] [--iterations 10] [--warmup 1] [--report timings.json]\n\n--cache-dir DIRECTORY or auto opts into a checksummed native disk cache for raw asset construction. Disabled by default; no Python cache protocol dependency.");
}
pub fn run(command: &str, opts: &BTreeMap<String, String>) -> Result<bool> {
    match command {
        "transform" => {
            let source = model(opts)?;
            let operations: Vec<Transform> = read_json(required(opts, "operations")?)?;
            let target = apply_pipeline(&source, &operations)?;
            let path = Path::new(required(opts, "output")?);
            parent(path)?;
            target.data.save(path, Some(&target.config))?;
            println!("{}", serde_json::to_string_pretty(&target.describe())?);
        }
        "precompute-rig" => {
            let store = AssetStore::new(required(opts, "assets")?);
            let archive = match required(opts, "rig")? {
                "anny" => {
                    let options = opts
                        .get("options")
                        .map(|s| read_json(s))
                        .transpose()?
                        .unwrap_or_default();
                    precompute_anny(&store, &options)?
                }
                "soma" => {
                    #[derive(serde::Deserialize)]
                    #[serde(default, deny_unknown_fields)]
                    struct Options {
                        threshold: f64,
                    }
                    impl Default for Options {
                        fn default() -> Self {
                            Self { threshold: 0.01 }
                        }
                    }
                    let o: Options = opts
                        .get("options")
                        .map(|s| read_json(s))
                        .transpose()?
                        .unwrap_or_default();
                    precompute_soma(&store, o.threshold)?
                }
                _ => return Err(error("precompute-rig requires --rig anny or soma")),
            };
            let path = Path::new(required(opts, "output")?);
            parent(path)?;
            archive.save(path)?;
            println!("Wrote native covariance cache {}", path.display());
        }
        "cache-orientations" => {
            let source = model(opts)?;
            let options = opts
                .get("options")
                .map(|s| read_json(s))
                .transpose()?
                .unwrap_or_default();
            let params = opts
                .get("params")
                .map(|s| read_json(s))
                .transpose()?
                .unwrap_or_default();
            let target = cache_model_orientations(&source, &params, &options)?;
            let path = Path::new(required(opts, "output")?);
            parent(path)?;
            target.data.save(path, Some(&target.config))?;
            println!(
                "Wrote native prepared model with cached orientations {}",
                path.display()
            );
        }
        "recompute-weights" => {
            let data = compute_cleaned_weights(&AssetStore::new(required(opts, "assets")?))?;
            write_json(required(opts, "output")?, &data)?;
            println!("Wrote cleaned weight JSON; source assets unchanged");
        }
        "export-calibration" => {
            let m = model(opts)?;
            let store = AssetStore::new(required(opts, "assets")?);
            let distribution = anny_core::distribution::SimpleShapeDistribution::load(&store, &m)?;
            write_json(required(opts, "output")?, &distribution)?;
        }
        "export-keypoints" => {
            let m = model(opts)?;
            let store = AssetStore::new(required(opts, "assets")?);
            let regressor = anny_core::tools::KeypointsRegressor::coco(&store, &m, None)?;
            write_json(required(opts, "output")?, &regressor)?;
        }
        "query" => {
            let m = model(opts)?;
            let request = read_json(required(opts, "request")?)?;
            let result = anny_core::operations::execute(&m, &request)?;
            if let Some(path) = opts.get("output") {
                write_json(path, &result)?;
            }
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        "benchmark" => {
            let iterations = number(opts, "iterations", 10)?;
            let warmup = number(opts, "warmup", 1)?;
            if iterations == 0 {
                return Err(error("benchmark iterations must be positive"));
            }
            let load = Instant::now();
            let m = model(opts)?;
            let load_seconds = load.elapsed().as_secs_f64();
            let p = opts
                .get("params")
                .map(|s| read_json(s))
                .transpose()?
                .unwrap_or_default();
            for _ in 0..warmup {
                std::hint::black_box(m.forward(&p)?);
            }
            let mut times = Vec::with_capacity(iterations);
            for _ in 0..iterations {
                let t = Instant::now();
                std::hint::black_box(m.forward(&p)?);
                times.push(t.elapsed().as_secs_f64());
            }
            let mut sorted = times.clone();
            sorted.sort_by(f64::total_cmp);
            let batch = m.forward(&p)?.get("vertices")?.shape[0];
            let report = json!({"schema":1,"backend":"cpu-f64","construction_or_load_seconds":load_seconds,"warmup":warmup,"iterations":iterations,"batch":batch,"vertices":m.data.vertex_count(),"bones":m.data.bone_count(),"seconds":times,"median_seconds":sorted[iterations/2],"p95_seconds":sorted[((iterations as f64*0.95).ceil() as usize-1).min(iterations-1)],"mean_seconds":times.iter().sum::<f64>()/iterations as f64,"input":"loaded native model; no per-iteration filesystem IO","os":std::env::consts::OS,"arch":std::env::consts::ARCH,"upstream":anny_core::UPSTREAM_REVISION});
            if let Some(path) = opts.get("report") {
                write_json(path, &report)?;
            }
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        _ => return Ok(false),
    }
    Ok(true)
}
pub fn fit_file(opts: &BTreeMap<String, String>) -> Result<()> {
    let m = model(opts)?;
    let target = anny_core::mesh_io::load(required(opts, "target")?)?;
    let mode: Correspondence = serde_json::from_value(json!(required(opts, "correspondence")?))?;
    let options: MeshFitOptions = opts
        .get("mesh-fit-options")
        .map(|s| read_json(s))
        .transpose()?
        .unwrap_or_default();
    if opts.contains_key("options") || opts.contains_key("inverter-options") {
        return Err(error(
            "mesh-file fitting uses --mesh-fit-options, not paired-vertex --options",
        ));
    }
    let result = fit_mesh(&m, &target, mode, &options)?;
    write_json(required(opts, "output")?, &result.fit.parameters)?;
    println!("{}", serde_json::to_string_pretty(&result.to_json())?);
    Ok(())
}
pub fn cached_model(store: AssetStore, config: AnnyConfig, dir: Option<&String>) -> Result<Anny> {
    let cache = match dir.map(String::as_str) {
        None => ModelCache::disabled(),
        Some("auto") => ModelCache::automatic()?,
        Some(path) => ModelCache::directory(path),
    };
    let result = cache.load_or_build(&store, &config)?;
    if let Some(p) = &result.path {
        eprintln!(
            "native cache {}: {}",
            if result.hit { "hit" } else { "miss" },
            p.display()
        );
    }
    Ok(result.model)
}
