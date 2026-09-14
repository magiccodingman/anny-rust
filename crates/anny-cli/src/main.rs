//! Thin command-line adapter; all model evaluation lives in anny-core.
mod authoring;
use anny_core::{
    assets::AssetStore,
    model::{Anny, Parameters},
    tensor::Archive,
    AnnyConfig, Error, Result, Tensor,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

fn error(s: impl Into<String>) -> Error {
    Error::Invalid(s.into())
}
fn read_json<T: serde::de::DeserializeOwned>(path: &str) -> Result<T> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}
fn parent(path: &Path) -> Result<()> {
    if let Some(p) = path.parent() {
        if !p.as_os_str().is_empty() {
            std::fs::create_dir_all(p)?;
        }
    }
    Ok(())
}
fn config(opts: &BTreeMap<String, String>) -> Result<Option<AnnyConfig>> {
    opts.get("config").map(|s| read_json(s)).transpose()
}
fn model(opts: &BTreeMap<String, String>) -> Result<Anny> {
    if opts.contains_key("model") && opts.contains_key("assets") {
        return Err(error("use --model OR --assets, not both"));
    }
    if let Some(path) = opts.get("model") {
        if opts.contains_key("cache-dir") {
            return Err(error("--cache-dir is only for asset construction"));
        }
        Anny::load(path, config(opts)?)
    } else {
        authoring::cached_model(
            AssetStore::new(opts.get("assets").cloned().unwrap_or_else(|| "data".into())),
            config(opts)?.unwrap_or_default(),
            opts.get("cache-dir"),
        )
    }
}
fn required<'a>(opts: &'a BTreeMap<String, String>, key: &str) -> Result<&'a str> {
    opts.get(key)
        .map(String::as_str)
        .ok_or_else(|| error(format!("missing --{key}")))
}
fn compare(opts: &BTreeMap<String, String>) -> Result<()> {
    let actual = Archive::load(required(opts, "actual")?)?;
    let expected = Archive::load(required(opts, "expected")?)?;
    let atol = opts.get("atol").map_or(Ok(1e-6), |s| {
        s.parse::<f64>().map_err(|_| error("invalid atol"))
    })?;
    let rtol = opts.get("rtol").map_or(Ok(0.), |s| {
        s.parse::<f64>().map_err(|_| error("invalid rtol"))
    })?;
    if !atol.is_finite() || !rtol.is_finite() || atol < 0. || rtol < 0. {
        return Err(error("tolerances must be finite and nonnegative"));
    }
    let mut failures = 0;
    let mut results = Vec::new();
    for (name, e) in &expected.tensors {
        let a = actual.tensor(name)?;
        if a.shape != e.shape {
            return Err(error(format!(
                "{name} shape mismatch {:?} vs {:?}",
                a.shape, e.shape
            )));
        }
        let mut max: f64 = 0.;
        let mut sq = 0.;
        let mut bad = 0;
        for (&a, &e) in a.data.iter().zip(&e.data) {
            let delta = (a - e).abs();
            max = max.max(delta);
            sq += delta * delta;
            let limit = if e.fract() == 0.
                && expected.tensors[name].kind != anny_core::tensor::Kind::Float
            {
                0.
            } else {
                atol + rtol * e.abs()
            };
            if !delta.is_finite() || delta > limit {
                bad += 1;
            }
        }
        failures += bad;
        results.push(serde_json::json!({"tensor":name,"shape":e.shape,"max_abs":max,"rms":(sq/e.data.len().max(1) as f64).sqrt(),"mismatches":bad}));
    }
    for key in [
        "bone_labels",
        "bone_parents",
        "blendshape_labels",
        "phenotype_labels",
        "local_change_labels",
        "facial_action_labels",
    ] {
        if let Some(e) = expected.metadata.get(key) {
            let a = actual
                .metadata
                .get(key)
                .ok_or_else(|| error(format!("missing metadata {key}")))?;
            let ev: serde_json::Value = serde_json::from_str(e)?;
            let av: serde_json::Value = serde_json::from_str(a)?;
            if ev != av {
                failures += 1;
                results.push(serde_json::json!({"metadata":key,"mismatches":1}));
            }
        }
    }
    let report =
        serde_json::json!({"passed":failures==0,"atol":atol,"rtol":rtol,"results":results});
    let text = serde_json::to_string_pretty(&report)?;
    if let Some(path) = opts.get("report") {
        parent(Path::new(path))?;
        std::fs::write(path, &text)?;
    }
    println!("{text}");
    if failures > 0 {
        return Err(error(format!("parity failed: {failures} differing values")));
    }
    Ok(())
}
fn run() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let command = args.next().unwrap_or_else(|| "help".into());
    if ["help", "--help", "-h"].contains(&command.as_str()) {
        println!("anny <prepare|generate|inspect|compare|measure|sample|fit|import-upstream|verify-assets> [options]\n\nimport-upstream --source UPSTREAM_CHECKOUT --destination NEW_DIRECTORY\n                [--allow-revision-mismatch true]\nverify-assets --assets data\nprepare --assets data [--config config.json] --output model.safetensors\ngenerate (--assets data | --model model.safetensors) [--config config.json]\n         [--params params.json] [--obj mesh.obj] [--output output.safetensors]\n         [--mesh character.glb] [--rigged true|false]\nlineup --assets data [--config config.json] --params lineup.json --output scene.glb\nmesh-convert --source input.obj --destination output.ply\ninspect (--assets data | --model model.safetensors) [--config config.json]\ncompare --actual output.safetensors --expected reference.safetensors\n        [--atol 0.000001] [--rtol 0] [--report report.json]\n\nmeasure (--assets data | --model model.safetensors) [--params params.json]\nsample --assets data [--config config.json] [--options sample-options.json] --output params.json\nfit (--assets data | --model model.safetensors) --target target.safetensors\n    [--options fit-options.json] [--inverter-options inverter-options.json] --output fitted.json\n\nInputs, matrices, and output arrays are row-major. OBJ uses upstream Z-up meters.");
        println!("generate/prepare/inspect/lineup/motion accept --precision f32|f64 (default f64). f32 uses native single-precision evaluation.");
        println!("motion (--assets data | --model prepared.safetensors) --source clip.json [--fps 30] [--precision f32|f64] --output motion.glb|clip.json\namass-inspect --source sequence.npz [--output description.json]");
        println!("amass-fit (--assets data | --model prepared.safetensors) --source sequence.npz --source-model supplied-smplx.safetensors --mapping explicit-map.json [--options fit.json] --output clip.json|clip.glb [--report fit-report.json]");
        authoring::help();
        return Ok(());
    }
    let allowed: BTreeSet<_> = [
        "source",
        "destination",
        "allow-revision-mismatch",
        "assets",
        "model",
        "config",
        "output",
        "params",
        "obj",
        "actual",
        "expected",
        "atol",
        "rtol",
        "report",
        "options",
        "inverter-options",
        "target",
        "mesh",
        "rigged",
        "cache-dir",
        "rig",
        "operations",
        "request",
        "mesh-fit-options",
        "correspondence",
        "iterations",
        "warmup",
        "precision",
        "fps",
        "source-model",
        "mapping",
    ]
    .into_iter()
    .collect();
    let mut opts = BTreeMap::new();
    while let Some(flag) = args.next() {
        let key = flag
            .strip_prefix("--")
            .ok_or_else(|| error(format!("unexpected argument {flag}")))?;
        if !allowed.contains(key) {
            return Err(error(format!("unknown flag {flag}")));
        }
        let value = args
            .next()
            .ok_or_else(|| error(format!("missing value for {flag}")))?;
        if opts.insert(key.into(), value).is_some() {
            return Err(error(format!("duplicate {flag}")));
        }
    }
    let single = match opts.get("precision").map(String::as_str) {
        None | Some("f64") => false,
        Some("f32") => true,
        _ => return Err(error("--precision expects f32 or f64")),
    };
    if opts.contains_key("precision")
        && !["generate", "prepare", "inspect", "lineup", "motion"].contains(&command.as_str())
    {
        return Err(error(
            "--precision currently applies to generate, prepare, inspect, lineup and motion",
        ));
    }
    if authoring::run(&command, &opts)? {
        return Ok(());
    }
    match command.as_str() {
        "amass-fit" => {
            let sequence = anny_core::motion::AmassSequence::from_npz(&std::fs::read(required(
                &opts, "source",
            )?)?)?;
            let source = anny_core::smpl::SmplModel::load(required(&opts, "source-model")?)?;
            let target = model(&opts)?;
            let mapping: anny_core::motion::VertexMap = read_json(required(&opts, "mapping")?)?;
            let options = opts
                .get("options")
                .map(|p| read_json(p))
                .transpose()?
                .unwrap_or_default();
            let result =
                anny_core::motion::fit_amass(&sequence, &source, &target, &mapping, &options)?;
            let dest = Path::new(required(&opts, "output")?);
            parent(dest)?;
            if dest.extension().and_then(|s| s.to_str()) == Some("json") {
                std::fs::write(dest, serde_json::to_vec_pretty(&result.clip)?)?;
            } else {
                result
                    .clip
                    .to_scene(&target, anny_core::motion::Precision::F64)?
                    .save(dest)?;
            }
            if let Some(report) = opts.get("report") {
                parent(Path::new(report))?;
                std::fs::write(report, serde_json::to_vec_pretty(&result)?)?;
            }
            println!(
                "Fitted {} frames with supplied source model and explicit correspondence",
                result.clip.frames.len()
            );
        }
        "amass-inspect" => {
            let sequence = anny_core::motion::AmassSequence::from_npz(&std::fs::read(required(
                &opts, "source",
            )?)?)?;
            let text = serde_json::to_string_pretty(&sequence.describe())?;
            if let Some(path) = opts.get("output") {
                parent(Path::new(path))?;
                std::fs::write(path, &text)?;
            }
            println!("{text}");
        }
        "motion" => {
            let model = model(&opts)?;
            let mut clip: anny_core::motion::PoseClip = read_json(required(&opts, "source")?)?;
            clip.validate(&model)?;
            if let Some(fps) = opts.get("fps") {
                clip = clip.resample(&model, fps.parse().map_err(|_| error("invalid fps"))?)?;
            }
            let dest = Path::new(required(&opts, "output")?);
            parent(dest)?;
            if dest.extension().and_then(|x| x.to_str()) == Some("json") {
                std::fs::write(dest, serde_json::to_vec_pretty(&clip)?)?;
            } else {
                let mode = if single {
                    anny_core::motion::Precision::F32
                } else {
                    anny_core::motion::Precision::F64
                };
                clip.to_scene(&model, mode)?.save(dest)?;
            }
            println!(
                "Exported {} native pose frames to {}",
                clip.frames.len(),
                dest.display()
            );
        }
        "import-upstream" => {
            let allow = opts
                .get("allow-revision-mismatch")
                .map(|s| s.parse::<bool>())
                .transpose()
                .map_err(|_| error("--allow-revision-mismatch expects true or false"))?
                .unwrap_or(false);
            let manifest = anny_core::import::import_upstream(
                required(&opts, "source")?,
                required(&opts, "destination")?,
                allow,
            )?;
            println!(
                "Imported and verified {} entries from {} without Python",
                manifest.files.len(),
                manifest.source_revision
            );
        }
        "verify-assets" => {
            let manifest = anny_core::import::verify(required(&opts, "assets")?)?;
            println!(
                "Verified {} entries; upstream {}",
                manifest.files.len(),
                manifest.source_revision
            );
        }
        "compare" => compare(&opts)?,
        "inspect" => {
            let m = model(&opts)?;
            let info = if single {
                m.to_f32()?.describe()
            } else {
                m.describe()
            };
            println!("{}", serde_json::to_string_pretty(&info)?);
        }
        "prepare" => {
            let m = model(&opts)?;
            let path = PathBuf::from(required(&opts, "output")?);
            parent(&path)?;
            if single {
                std::fs::write(&path, m.to_f32()?.to_bytes()?)?;
            } else {
                m.data.save(&path, Some(&m.config))?;
            }
            println!(
                "Prepared {} vertices, {} bones: {}",
                m.data.vertex_count(),
                m.data.bone_count(),
                path.display()
            );
        }
        "measure" => {
            let m = model(&opts)?;
            let params: Parameters = opts
                .get("params")
                .map(|p| read_json(p))
                .transpose()?
                .unwrap_or_default();
            let result = m.forward(&params)?;
            let measure =
                anny_core::tools::Anthropometry::new(&m)?.measure(result.get("rest_vertices")?)?;
            println!("{}", serde_json::to_string_pretty(&measure)?);
        }
        "sample" => {
            let m = model(&opts)?;
            let store = AssetStore::new(required(&opts, "assets")?);
            let distribution = anny_core::distribution::SimpleShapeDistribution::load(&store, &m)?;
            let options = opts
                .get("options")
                .map(|p| read_json(p))
                .transpose()?
                .unwrap_or_default();
            let result = distribution.sample(&options)?;
            let path = Path::new(required(&opts, "output")?);
            parent(path)?;
            std::fs::write(path, serde_json::to_vec_pretty(&result.parameters)?)?;
            println!(
                "Sampled {} character(s); chronological ages: {:?}",
                result.morphological_age.len(),
                result.morphological_age
            );
        }
        "fit" => {
            if !required(&opts, "target")?.ends_with(".safetensors") {
                return authoring::fit_file(&opts);
            }
            if opts.contains_key("correspondence") || opts.contains_key("mesh-fit-options") {
                return Err(error("Safetensors fit already uses paired vertex correspondence; use --options/--inverter-options"));
            }
            let m = model(&opts)?;
            let target = Archive::load(required(&opts, "target")?)?;
            let setup = opts
                .get("inverter-options")
                .map(|p| read_json(p))
                .transpose()?
                .unwrap_or_default();
            let options = opts
                .get("options")
                .map(|p| read_json(p))
                .transpose()?
                .unwrap_or_default();
            let fitter = anny_core::inverter::AnnyInverter::new(&m, setup)?;
            let result = fitter.fit(target.tensor("vertices")?, &options)?;
            let path = Path::new(required(&opts, "output")?);
            parent(path)?;
            std::fs::write(path, serde_json::to_vec_pretty(&result.parameters)?)?;
            println!("{}", serde_json::to_string_pretty(&result.to_json())?);
        }
        "mesh-convert" => {
            let input = required(&opts, "source")?;
            let destination = Path::new(required(&opts, "destination")?);
            let mesh = anny_core::mesh_io::load(input)?;
            parent(destination)?;
            match destination
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_ascii_lowercase()
                .as_str()
            {
                "glb" | "gltf" => {
                    let mut scene = anny_core::scene::Scene::new();
                    scene.objects.push(anny_core::scene::SceneObject {
                        name: "Imported surface".into(), mesh,
                        transform: anny_core::math::Mat4::identity(), color: [0.7,0.7,0.7,1.], skin: None,
                        extras: serde_json::json!({"note":"geometry-only conversion; skin/default morph pose baked on import"}),
                    });
                    scene.save(destination)?;
                }
                _ => anny_core::mesh_io::save(&mesh, destination)?,
            }
            println!("Converted geometry to {}", destination.display());
        }
        "lineup" => {
            #[derive(serde::Deserialize)]
            #[serde(deny_unknown_fields)]
            struct Character {
                #[serde(default)]
                parameters: Parameters,
                #[serde(default)]
                export: anny_core::scene::CharacterExport,
            }
            let characters: Vec<Character> = read_json(required(&opts, "params")?)?;
            let m = model(&opts)?;
            let mut scene = anny_core::scene::Scene::new();
            let m32 = if single { Some(m.to_f32()?) } else { None };
            for character in characters {
                let output = if let Some(m32) = &m32 {
                    m32.forward(&character.parameters)?.to_reference()
                } else {
                    m.forward(&character.parameters)?
                };
                scene.add_evaluated_character(
                    &m,
                    &character.parameters,
                    &character.export,
                    &output,
                )?;
            }
            let path = Path::new(required(&opts, "output")?);
            parent(path)?;
            scene.save(path)?;
            println!(
                "Exported {} separate characters to {}",
                scene.objects.len(),
                path.display()
            );
        }
        "generate" => {
            if !opts.contains_key("output")
                && !opts.contains_key("obj")
                && !opts.contains_key("mesh")
            {
                return Err(error("generate needs --output, --obj, or --mesh"));
            }
            let m = model(&opts)?;
            let params: Parameters = opts
                .get("params")
                .map(|p| read_json(p))
                .transpose()?
                .unwrap_or_default();
            let out = if single {
                m.to_f32()?.forward(&params)?.to_reference()
            } else {
                m.forward(&params)?
            };
            if let Some(path) = opts.get("mesh") {
                let mut scene = anny_core::scene::Scene::new();
                let rigged = match opts.get("rigged").map(String::as_str) {
                    None | Some("false") => false,
                    Some("true") => true,
                    _ => return Err(error("--rigged expects true or false")),
                };
                scene.add_evaluated_character(
                    &m,
                    &params,
                    &anny_core::scene::CharacterExport {
                        rigged,
                        ..Default::default()
                    },
                    &out,
                )?;
                parent(Path::new(path))?;
                match Path::new(path)
                    .extension()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_ascii_lowercase()
                    .as_str()
                {
                    "glb" | "gltf" => scene.save(path)?,
                    _ if rigged => {
                        return Err(error(
                            "rigged export requires GLB/glTF; use --rigged false to bake a surface",
                        ))
                    }
                    _ => anny_core::mesh_io::save(&scene.objects[0].mesh, path)?,
                }
            }
            if let Some(path) = opts.get("obj") {
                let v = out.get("vertices")?;
                if v.shape[0] != 1 {
                    return Err(error(
                        "OBJ export supports one mesh; use --output for batches",
                    ));
                }
                let vertices = Tensor::new(vec![v.shape[1], 3], v.data.clone())?;
                parent(Path::new(path))?;
                let uv = m
                    .data
                    .arrays
                    .get("texture_coordinates")
                    .zip(m.data.arrays.get("face_texture_coordinate_indices"));
                anny_core::mesh::save_obj(path, &vertices, m.data.get("faces")?, uv)?;
            }
            if let Some(path) = opts.get("output") {
                parent(Path::new(path))?;
                if path.ends_with(".json") {
                    std::fs::write(path, serde_json::to_vec(&out.to_json())?)?;
                } else {
                    let mut a = Archive {
                        tensors: out.arrays,
                        metadata: Default::default(),
                    };
                    for key in [
                        "template_vertices",
                        "faces",
                        "texture_coordinates",
                        "face_texture_coordinate_indices",
                        "vertex_bone_indices",
                        "vertex_bone_weights",
                        "base_mesh_vertex_indices",
                    ] {
                        if let Some(t) = m.data.arrays.get(key) {
                            a.tensors.insert(key.into(), t.clone());
                        }
                    }
                    let info = m.describe();
                    for key in [
                        "bone_labels",
                        "bone_parents",
                        "blendshape_labels",
                        "phenotype_labels",
                        "local_change_labels",
                        "facial_action_labels",
                    ] {
                        let value = if key == "blendshape_labels" {
                            serde_json::to_value(&m.data.metadata.blendshape_labels)?
                        } else {
                            info[key].clone()
                        };
                        a.metadata
                            .insert(key.into(), serde_json::to_string(&value)?);
                    }
                    a.metadata.insert(
                        "anny_rust_evaluation_precision".into(),
                        if single { "f32" } else { "f64" }.into(),
                    );
                    if single {
                        a.save_f32(path)?;
                    } else {
                        a.save(path)?;
                    }
                }
            }
            println!(
                "Generated {} vertices, {} bones",
                m.data.vertex_count(),
                m.data.bone_count()
            );
        }
        _ => return Err(error(format!("unknown command {command}; use anny help"))),
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("anny: {e}");
        std::process::exit(1);
    }
}
