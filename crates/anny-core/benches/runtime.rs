//! Native runtime benchmarks.
//!
//! Deliberately dependency-free (`harness = false`): the numbers that matter for this project are
//! absolute wall-clock costs of real operations on the committed model, and a self-contained harness
//! keeps `cargo bench` working offline and keeps `Cargo.lock` untouched.
//!
//! ```text
//! cargo bench -p anny-core --bench runtime
//! ```
//!
//! Real assets are required. Without them the harness reports that it cannot run and exits
//! successfully, so `cargo bench` never fails on a checkout without `data/`.

use anny_core::{
    assets::AssetStore,
    config::*,
    math::write4,
    model::identity_poses,
    operations::{execute, Request},
    Parameters, Result,
};
use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn data_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data")
}

struct Stats {
    min: Duration,
    median: Duration,
}

/// Run `f` `iters` times after a warmup, returning min and median frame times.
fn measure(iters: u32, warmup: u32, mut f: impl FnMut() -> Result<()>) -> Result<Stats> {
    for _ in 0..warmup {
        f()?;
    }
    let mut samples = Vec::with_capacity(iters as usize);
    for _ in 0..iters {
        let start = Instant::now();
        f()?;
        samples.push(start.elapsed());
    }
    samples.sort();
    Ok(Stats {
        min: samples[0],
        median: samples[samples.len() / 2],
    })
}

fn report(group: &str, name: &str, iters: u32, stats: &Stats) {
    println!(
        "{group:<12} {name:<34} {iters:>4} iters  min {:>10.3} ms  median {:>10.3} ms",
        stats.min.as_secs_f64() * 1e3,
        stats.median.as_secs_f64() * 1e3
    );
}

fn batch_poses(batch: usize, bones: usize) -> anny_core::Tensor {
    let mut pose = identity_poses(batch, bones);
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as f64 / ((1u64 << 31) as f64) - 1.0
    };
    for bone in 0..bones {
        for item in 0..batch {
            let start = (item * bones + bone) * 16;
            let rotation = anny_core::math::Vec3::new(next() * 0.05, next() * 0.05, next() * 0.05);
            write4(
                &anny_core::math::rigid(&anny_core::math::rotvec(&rotation), &Default::default()),
                &mut pose.data[start..start + 16],
            );
        }
    }
    pose
}

/// One character, a different pose every call: the animating/editor-slider workload. The pose has to
/// change or the compiler could hoist the update out of the measurement loop.
fn editor_pose(bones: usize, tick: u32) -> serde_json::Value {
    let mut pose = identity_poses(1, bones);
    let phase = tick as f64 * 0.05;
    write4(
        &anny_core::math::rigid(
            &anny_core::math::rotvec(&anny_core::math::Vec3::new(0.0, phase, 0.0)),
            &anny_core::math::Vec3::new(0.0, 0.0, 0.0),
        ),
        &mut pose.data[..16],
    );
    write4(
        &anny_core::math::rigid(
            &anny_core::math::rotvec(&anny_core::math::Vec3::new(
                phase * 0.5,
                -phase * 0.25,
                phase * 0.75,
            )),
            &Default::default(),
        ),
        &mut pose.data[16..32],
    );
    pose.nested_json()
}

fn main() -> Result<()> {
    let root = data_root();
    if !root.exists() {
        println!(
            "assets not present at {}; nothing to benchmark",
            root.display()
        );
        return Ok(());
    }
    let store = AssetStore::new(&root);
    println!(
        "anny-core native runtime benchmarks\ndata root: {}\n",
        root.display()
    );

    let default = AnnyConfig::default();
    let prepared = store.build(&default)?;
    let bones = prepared.data.bone_count();
    println!(
        "default model: {} vertices, {} faces, {} bones\n",
        prepared.data.vertex_count(),
        prepared.data.get("faces")?.shape[0],
        bones
    );

    // Cold prepare: loads and converts the committed upstream archives, then evaluates the model.
    let stats = measure(3, 0, || {
        let model = store.build(&default)?;
        black_box(model.data.vertex_count());
        Ok(())
    })?;
    report("prepare", "prepare default (cold)", 3, &stats);

    // Prepared-model reload: the in-memory path an application takes at startup.
    let typed_prepared = prepared.to_f32()?;
    let bytes = typed_prepared.to_bytes()?;
    println!(
        "prepared f32 payload: {:.1} MB\n",
        bytes.len() as f64 / (1024.0 * 1024.0)
    );
    let stats = measure(5, 1, || {
        let model = anny_core::AnnyF32::from_bytes(&bytes, Some(default.clone()))?;
        black_box(model.data().vertex_count());
        Ok(())
    })?;
    report("prepare", "reload prepared f32 bytes", 5, &stats);

    // Forward evaluation across the configurations that matter for games and tools.
    let configurations = vec![
        ("f64 default", default.clone()),
        ("f64 dqs", {
            let mut c = default.clone();
            c.skinning_method = SkinningMethod::Dqs;
            c
        }),
        ("f64 makehuman rig", {
            let mut c = default.clone();
            c.rig = RigSpec::Name("makehuman".into());
            c
        }),
        ("f64 all phenotypes", {
            let mut c = default.clone();
            c.phenotypes = "all".into();
            c
        }),
        ("f64 local+facial all", {
            let mut c = default.clone();
            c.local_changes = Selection::all();
            c.facial_actions = Selection::all();
            c
        }),
    ];
    for (name, config) in &configurations {
        let model = store.build(config)?;
        let parameters = anny_core::Parameters {
            phenotype_kwargs: serde_json::json!({"gender": 0.61, "age": 0.42, "height": 0.57}),
            ..Default::default()
        };
        let stats = measure(20, 3, || {
            black_box(model.forward(&parameters)?);
            Ok(())
        })?;
        report("generate", name, 20, &stats);
    }

    // The real f32 runtime path: converted once, then evaluated through the typed API. These are
    // not f64 evaluations with a cast on the output.
    for (name, method) in [
        ("f32 typed (lbs)", SkinningMethod::Lbs),
        ("f32 typed (dqs)", SkinningMethod::Dqs),
        ("f32 typed (warp_lbs)", SkinningMethod::WarpLbs),
    ] {
        let mut config = default.clone();
        config.skinning_method = method;
        let typed = store.build(&config)?.to_f32()?;
        let parameters = anny_core::Parameters {
            phenotype_kwargs: serde_json::json!({"gender": 0.61, "age": 0.42, "height": 0.57}),
            ..Default::default()
        };
        let stats = measure(20, 3, || {
            black_box(typed.forward(&parameters)?);
            Ok(())
        })?;
        report("generate", name, 20, &stats);
    }

    // Batch generation: the population workflow.
    for batch in [1, 10, 100] {
        let parameters = anny_core::Parameters {
            pose_parameters: batch_poses(batch, bones).nested_json(),
            ..Default::default()
        };
        // The batch dimension has to actually be the number of characters, otherwise these
        // numbers would silently describe a single evaluation.
        let vertices = prepared.forward(&parameters)?.get("vertices")?.clone();
        assert_eq!(
            vertices.shape,
            vec![batch, prepared.data.vertex_count(), 3],
            "batch generation did not produce {batch} characters"
        );
        let iters = if batch >= 100 { 5 } else { 10 };
        let stats = measure(iters, 1, || {
            black_box(prepared.forward(&parameters)?);
            Ok(())
        })?;
        report("batch", &format!("generate x{batch}"), iters, &stats);
        println!(
            "{:<12} {:<34} {:>4}       marginal {:>10.3} us/character",
            "",
            "",
            "",
            stats.min.as_secs_f64() * 1e6 / batch as f64
        );
    }

    // Where the fixed per-call cost actually goes. `forward` is coefficients + rest model + pose
    // model; the first two do not depend on the pose, so a pose session caches them. Measuring the
    // three steps separately is what makes the session claim falsifiable rather than asserted.
    let coefficients = prepared.coefficients(&Parameters::default())?;
    let stats = measure(20, 3, || {
        black_box(prepared.coefficients(&Parameters::default())?);
        Ok(())
    })?;
    report("split", "coefficients default", 20, &stats);

    let stats = measure(20, 3, || {
        black_box(prepared.rest_model(&coefficients)?);
        Ok(())
    })?;
    report("split", "rest_model default", 20, &stats);

    // The repeated-update path: one session, a fresh pose every iteration. This is the animating
    // character / editor slider case, and the pose differs each time so it cannot be hoisted.
    let mut session = prepared.pose_session(&Parameters::default())?;
    let mut tick = 0u32;
    let stats = measure(50, 3, || {
        tick += 1;
        black_box(session.update(&editor_pose(bones, tick))?);
        Ok(())
    })?;
    report("session", "update pose (reused rest)", 50, &stats);

    // Building the session is not free, so it has to be shown next to the per-update cost: the
    // break-even point is what decides whether a caller should use a session at all.
    let stats = measure(10, 2, || {
        black_box(prepared.pose_session(&Parameters::default())?);
        Ok(())
    })?;
    report("session", "build (coefficients + rest)", 10, &stats);

    // Secondary operations through the shared control-plane API.
    let regressor = anny_core::tools::KeypointsRegressor::coco(&store, &prepared, None)?;
    println!("\nkeypoint regressor: {} labels\n", regressor.labels.len());
    let requests: Vec<(&str, Request)> = vec![
        (
            "measure",
            Request::Measure {
                parameters: Default::default(),
            },
        ),
        (
            "keypoints",
            Request::Keypoints {
                parameters: Default::default(),
                regressor: regressor.clone(),
            },
        ),
        (
            "collision",
            Request::Collision {
                parameters: Default::default(),
                group_toes: true,
                group_eyes: true,
                group_tongue: true,
            },
        ),
        (
            "pose-convert world",
            Request::PoseConvert {
                parameters: Default::default(),
                mode: PoseParameterization::World,
            },
        ),
    ];
    for (name, request) in &requests {
        let stats = match measure(10, 2, || {
            black_box(execute(&prepared, request)?);
            Ok(())
        }) {
            Ok(stats) => stats,
            Err(error) => {
                println!("{:<12} {name:<34} skipped: {error}", "derive");
                continue;
            }
        };
        report("derive", name, 10, &stats);
    }

    Ok(())
}
