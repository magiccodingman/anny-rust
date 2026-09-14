//! Opt-in byte-level checks against the Python-converted assets committed in data/.
use anny_core::{torch_archive, Result};
use safetensors::SafeTensors;
use std::path::Path;
fn visit(path: &Path, files: &mut Vec<std::path::PathBuf>) {
    for e in std::fs::read_dir(path).unwrap() {
        let e = e.unwrap();
        if e.file_type().unwrap().is_dir() {
            visit(&e.path(), files);
        } else if matches!(
            e.path().extension().and_then(|s| s.to_str()),
            Some("pt" | "pth")
        ) {
            files.push(e.path());
        }
    }
}
#[test]
#[ignore = "real committed asset comparison; run cargo test -p anny-core --release --test native_import -- --ignored"]
fn all_committed_tensor_archives_match_python_conversion() -> Result<()> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
    let mut files = vec![];
    visit(&root, &mut files);
    assert_eq!(files.len(), 8);
    for file in files {
        let converted = torch_archive::convert(&std::fs::read(&file)?)?;
        let expected = std::fs::read(format!("{}.safetensors", file.display()))?;
        let a = SafeTensors::deserialize(&converted)?;
        let e = SafeTensors::deserialize(&expected)?;
        assert_eq!(a.len(), e.len(), "{}", file.display());
        for (name, t) in e.iter() {
            let other = a.tensor(name)?;
            assert_eq!(other.dtype(), t.dtype(), "{}: {name} dtype", file.display());
            assert_eq!(other.shape(), t.shape(), "{}: {name} shape", file.display());
            assert_eq!(other.data(), t.data(), "{}: {name} values", file.display());
        }
        let (_, a) = SafeTensors::read_metadata(&converted)?;
        let (_, e) = SafeTensors::read_metadata(&expected)?;
        let a = a.metadata().as_ref().unwrap();
        let e = e.metadata().as_ref().unwrap();
        assert_eq!(a["source_sha256"], e["source_sha256"]);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&a["anny_port_payload"])?,
            serde_json::from_str::<serde_json::Value>(&e["anny_port_payload"])?
        );
        eprintln!("PASS {}", file.display());
    }
    Ok(())
}
