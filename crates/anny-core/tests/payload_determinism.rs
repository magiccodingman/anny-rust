//! A written payload has to be byte-identical across processes.
//!
//! `safetensors` serializes `__metadata__` straight out of a `HashMap`, and `std` seeds a `HashMap`'s
//! hasher per process, so the same build used to write a different file in every process. No single
//! process can observe that: two serializations inside one process agree even at the old code. So
//! this test re-runs itself as a child process twice and compares whole-file digests — three
//! processes, one file — for both writers `prepare` can take (`Archive::to_bytes` for f64 and
//! `ArchiveF32::to_bytes` for f32).
use anny_core::{assets::AssetStore, config::AnnyConfig, *};
use sha2::{Digest, Sha256};
use std::process::Command;

/// Set in the spawned child so it computes and prints instead of spawning its own children.
const CHILD: &str = "ANNY_PAYLOAD_DETERMINISM_CHILD";

fn digest(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Whole-file digests of both prepared payloads. The f64 path is what `prepare` writes by default
/// (`ModelData::save` -> `archive(config)` -> `Archive::to_bytes`); the f32 path is
/// `ModelData::to_f32()?.to_bytes()`.
fn digests() -> Result<Vec<(&'static str, String)>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let config = AnnyConfig::default();
    let data = AssetStore::new(root.join("data")).build(&config)?;
    let double = data.data.archive(Some(&data.config))?.to_bytes()?;
    let single = data.to_f32()?.to_bytes()?;
    Ok(vec![("f64", digest(&double)), ("f32", digest(&single))])
}

fn parse(text: &str, name: &str) -> Option<String> {
    let prefix = format!("payload digest {name}: ");
    text.lines()
        .find_map(|line| line.strip_prefix(&prefix))
        .map(|d| d.trim().to_owned())
}

/// The child half. Cheap and inert when not spawned, so it can stay a normal test.
#[test]
fn payload_digest_child() -> Result<()> {
    if std::env::var(CHILD).is_err() {
        return Ok(());
    }
    for (name, value) in digests()? {
        println!("payload digest {name}: {value}");
    }
    Ok(())
}

#[test]
#[ignore = "real committed asset comparison; run cargo test -p anny-core --release --test payload_determinism -- --include-ignored"]
fn payload_is_byte_identical_across_processes() -> Result<()> {
    let mine = digests()?;
    let exe = std::env::current_exe()?;
    for round in 0..2 {
        let out = Command::new(&exe)
            .args(["--exact", "payload_digest_child", "--nocapture"])
            .env(CHILD, "1")
            .output()?;
        let text = String::from_utf8_lossy(&out.stdout);
        for (name, expected) in &mine {
            let actual = parse(&text, name).unwrap_or_else(|| {
                panic!("child {round} printed no {name} digest; stdout was:\n{text}")
            });
            assert_eq!(
                &actual, expected,
                "the {name} payload the same build writes differs between processes"
            );
        }
    }
    Ok(())
}
