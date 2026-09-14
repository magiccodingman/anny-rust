//! Native source import and verification. No Python interpreter is required.
use crate::{ensure, torch_archive, Error, Result, UPSTREAM_REVISION};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImportRecord {
    pub path: String,
    pub sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub derived_from: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImportManifest {
    pub schema: u32,
    pub source_revision: String,
    pub expected_revision: String,
    pub files: Vec<ImportRecord>,
}
pub fn hash_file(path: &Path) -> Result<String> {
    let mut f = fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut b = [0u8; 65536];
    loop {
        let n = f.read(&mut b)?;
        if n == 0 {
            break;
        }
        h.update(&b[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}
fn list(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
    for e in fs::read_dir(dir)? {
        let e = e?;
        let t = e.file_type()?;
        ensure(
            !t.is_symlink(),
            format!(
                "symlink not allowed in imported assets: {}",
                e.path().display()
            ),
        )?;
        if e.file_name() == "__pycache__" {
            continue;
        }
        if t.is_dir() {
            list(&e.path(), files)?;
        } else if t.is_file() {
            files.push(e.path());
        } else {
            return Err(Error::Invalid("non-regular source asset".into()));
        }
    }
    Ok(())
}
fn relative(path: &Path, root: &Path) -> Result<String> {
    Ok(path
        .strip_prefix(root)
        .map_err(|_| Error::Invalid("asset outside source".into()))?
        .to_string_lossy()
        .replace('\\', "/"))
}
/// Read-only manifest verification. Reports missing/mismatched assets as errors.
pub fn verify(root: impl AsRef<Path>) -> Result<ImportManifest> {
    let root = root.as_ref().canonicalize()?;
    let manifest: ImportManifest =
        serde_json::from_slice(&fs::read(root.join("import-manifest.json"))?)?;
    ensure(manifest.schema == 1, "unsupported import manifest schema")?;
    let mut seen = std::collections::BTreeSet::new();
    for r in &manifest.files {
        let p = Path::new(&r.path);
        ensure(
            !p.is_absolute()
                && p.components()
                    .all(|c| matches!(c, std::path::Component::Normal(_))),
            "unsafe manifest path",
        )?;
        ensure(seen.insert(&r.path), "duplicate manifest path")?;
        let file = root.join(p);
        let canonical = file.canonicalize()?;
        ensure(
            canonical.starts_with(&root),
            "manifest path escapes data directory",
        )?;
        ensure(
            hash_file(&file)? == r.sha256,
            format!("asset checksum mismatch: {}", r.path),
        )?;
        if let Some(n) = r.bytes {
            ensure(
                fs::metadata(file)?.len() == n,
                format!("asset size mismatch: {}", r.path),
            )?;
        }
    }
    Ok(manifest)
}
/// Import a pinned checkout to a NEW directory, committing it only after every
/// conversion succeeds. The source tree is never mutated; an existing output is
/// refused so neither user files nor already-imported data can be overwritten.
pub fn import_upstream(
    source: impl AsRef<Path>,
    destination: impl AsRef<Path>,
    allow_revision_mismatch: bool,
) -> Result<ImportManifest> {
    let source = source.as_ref().canonicalize()?;
    let data = source.join("src/anny/data").canonicalize()?;
    ensure(
        data.join("mpfb2/3dobjs/base.obj").is_file(),
        "not an Anny source checkout",
    )?;
    let revision = std::process::Command::new("git")
        .args(["-C"])
        .arg(&source)
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_owned())
        .unwrap_or_else(|| "unknown".into());
    ensure(allow_revision_mismatch || revision==UPSTREAM_REVISION,format!("upstream revision {revision}; expected {UPSTREAM_REVISION}. Use a pinned worktree or explicit --allow-revision-mismatch"))?;
    if !allow_revision_mismatch {
        let status = std::process::Command::new("git")
            .arg("-C")
            .arg(&source)
            .args([
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                "src/anny/data",
            ])
            .output()?;
        ensure(status.status.success() && status.stdout.is_empty(),
            "source assets have local changes; use a clean pinned worktree or explicit --allow-revision-mismatch")?;
    }
    let destination = destination.as_ref();
    ensure(!destination.exists(),"import destination already exists; use a fresh directory (existing assets are never overwritten)")?;
    let parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let parent = parent.canonicalize()?;
    ensure(
        !parent.starts_with(&data),
        "destination overlaps upstream assets",
    )?;
    let name = destination
        .file_name()
        .ok_or_else(|| Error::Invalid("missing destination name".into()))?;
    let dest = parent.join(name);
    let staging = parent.join(format!(
        ".anny-import-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    fs::create_dir(&staging)?;
    let result = (|| {
        let mut sources = vec![];
        list(&data, &mut sources)?;
        sources.sort();
        let mut records = vec![];
        for src in sources {
            let rel = relative(&src, &data)?;
            let dst = staging.join(&rel);
            fs::create_dir_all(dst.parent().unwrap())?;
            fs::copy(&src, &dst)?;
            let ext = src.extension().and_then(|s| s.to_str()).unwrap_or("");
            let converted = match ext {
                "pt" | "pth" => Some((
                    format!("{rel}.safetensors"),
                    torch_archive::convert(&fs::read(&src)?)?,
                )),
                "yaml" | "yml" => {
                    let value: serde_json::Value = serde_yaml::from_slice(&fs::read(&src)?)
                        .map_err(|e| Error::Invalid(format!("YAML {rel}: {e}")))?;
                    Some((format!("{rel}.json"), serde_json::to_vec_pretty(&value)?))
                }
                _ => None,
            };
            if let Some((path, bytes)) = converted {
                let file = staging.join(&path);
                fs::write(&file, bytes)?;
                records.push(ImportRecord {
                    path,
                    sha256: hash_file(&file)?,
                    bytes: Some(fs::metadata(file)?.len()),
                    derived_from: Some(rel.clone()),
                });
            }
            records.push(ImportRecord {
                path: rel,
                sha256: hash_file(&dst)?,
                bytes: Some(fs::metadata(dst)?.len()),
                derived_from: None,
            });
        }
        let manifest = ImportManifest {
            schema: 1,
            source_revision: revision,
            expected_revision: UPSTREAM_REVISION.into(),
            files: records,
        };
        fs::write(
            staging.join("import-manifest.json"),
            serde_json::to_vec_pretty(&manifest)?,
        )?;
        verify(&staging)?;
        ensure(!dest.exists(), "destination appeared while importing")?;
        fs::rename(&staging, &dest)?;
        Ok(manifest)
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(staging);
    }
    result
}
