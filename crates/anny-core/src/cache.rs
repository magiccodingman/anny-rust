//! Optional native prepared-model cache. Content hashes, not timestamps or a
//! mutable import manifest, identify source assets. Cache files are atomically
//! published and checksum-verified before deserialization. This is a Rust cache
//! protocol, not a claim of Python's cache filename compatibility.
use crate::{assets::AssetStore, ensure, Anny, AnnyConfig, Error, Result};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};
const MAGIC: &[u8; 8] = b"ANNYCAC1";
const HEADER: usize = 80;
const MAX_CACHE_BYTES: u64 = 4 * 1024 * 1024 * 1024;
static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheKey(String);
impl CacheKey {
    pub fn as_str(&self) -> &str {
        &self.0
    }
    /// A caller supplying its own asset loader must hash every dependency into
    /// asset_digest. AssetStore callers should use ModelCache::load_or_build.
    pub fn for_config(config: &AnnyConfig, asset_digest: &str) -> Result<Self> {
        config.validate()?;
        ensure(
            asset_digest.len() == 64 && asset_digest.bytes().all(|c| c.is_ascii_hexdigit()),
            "asset digest must be a SHA-256 hex string",
        )?;
        fn canonical(v: serde_json::Value) -> serde_json::Value {
            match v {
                serde_json::Value::Object(m) => {
                    let sorted: std::collections::BTreeMap<_, _> = m.into_iter().collect();
                    serde_json::Value::Object(
                        sorted.into_iter().map(|(k, v)| (k, canonical(v))).collect(),
                    )
                }
                serde_json::Value::Array(v) => {
                    serde_json::Value::Array(v.into_iter().map(canonical).collect())
                }
                v => v,
            }
        }
        let mut h = Sha256::new();
        h.update(b"anny-rust/prepared-cache-v1\0");
        h.update(crate::UPSTREAM_REVISION.as_bytes());
        h.update((crate::DATA_VERSION as u64).to_le_bytes());
        h.update(asset_digest.to_ascii_lowercase().as_bytes());
        h.update(serde_json::to_vec(&canonical(serde_json::to_value(
            config,
        )?))?);
        Ok(Self(format!("{:x}", h.finalize())))
    }
}
#[derive(Clone, Debug, Default)]
pub struct ModelCache {
    pub directory: Option<PathBuf>,
}
pub struct CachedModel {
    pub model: Anny,
    pub hit: bool,
    pub path: Option<PathBuf>,
}
impl ModelCache {
    pub fn disabled() -> Self {
        Self::default()
    }
    pub fn directory(path: impl Into<PathBuf>) -> Self {
        Self {
            directory: Some(path.into()),
        }
    }
    /// Explicit opt-in to an OS-appropriate user cache, honoring ANNY_CACHE_DIR.
    pub fn automatic() -> Result<Self> {
        if let Some(p) = std::env::var_os("ANNY_CACHE_DIR").filter(|p| !p.is_empty()) {
            return Ok(Self::directory(p));
        }
        if cfg!(target_arch = "wasm32") {
            return Err(Error::Invalid(
                "filesystem cache unavailable in the browser; use prepared bytes".into(),
            ));
        }
        let root = if cfg!(target_os = "windows") {
            std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
        } else if cfg!(target_os = "macos") {
            std::env::var_os("HOME").map(|p| PathBuf::from(p).join("Library/Caches"))
        } else {
            std::env::var_os("XDG_CACHE_HOME")
                .filter(|p| !p.is_empty())
                .map(PathBuf::from)
                .or_else(|| std::env::var_os("HOME").map(|p| PathBuf::from(p).join(".cache")))
        }
        .ok_or_else(|| Error::Invalid("no user cache directory; supply an explicit path".into()))?;
        Ok(Self::directory(root.join("anny-rust")))
    }
    pub fn load_or_build(&self, store: &AssetStore, config: &AnnyConfig) -> Result<CachedModel> {
        if let Some(dir) = &self.directory {
            let root = store.root.canonicalize()?;
            let mut ancestor = dir.as_path();
            while !ancestor.exists() {
                ancestor = ancestor
                    .parent()
                    .filter(|p| !p.as_os_str().is_empty())
                    .unwrap_or(Path::new("."));
            }
            ensure(
                !ancestor.canonicalize()?.starts_with(&root),
                "cache directory must be outside the source assets",
            )?;
            let digest = asset_fingerprint(store, config)?;
            let key = CacheKey::for_config(config, &digest)?;
            self.load_or_build_with(&key, || {
                let model = store.build(config)?;
                ensure(
                    digest == asset_fingerprint(store, config)?,
                    "assets changed during construction; cache was not published",
                )?;
                Ok(model)
            })
        } else {
            Ok(CachedModel {
                model: store.build(config)?,
                hit: false,
                path: None,
            })
        }
    }
    /// Useful with custom source providers. The caller owns dependency discovery
    /// when using this lower-level entry point.
    pub fn load_or_build_with(
        &self,
        key: &CacheKey,
        build: impl FnOnce() -> Result<Anny>,
    ) -> Result<CachedModel> {
        let Some(dir) = &self.directory else {
            return Ok(CachedModel {
                model: build()?,
                hit: false,
                path: None,
            });
        };
        fs::create_dir_all(dir)?;
        let path = dir.join(format!("{}.annycache", key.0));
        if path.exists() {
            return Ok(CachedModel {
                model: read_cache(&path, key)?,
                hit: true,
                path: Some(path),
            });
        }
        let model = build()?;
        let bytes = model.data.archive(Some(&model.config))?.to_bytes()?;
        ensure(
            bytes.len() as u64 <= MAX_CACHE_BYTES,
            "prepared cache exceeds size limit",
        )?;
        let mut header = Vec::with_capacity(HEADER);
        header.extend_from_slice(MAGIC);
        header.extend_from_slice(key.0.as_bytes()); // fixed 64 ASCII bytes
        header.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        let checksum = Sha256::digest(&bytes);
        // A new, same-directory temporary file prevents partial-reader exposure.
        let mut selected = None;
        for _ in 0..32 {
            let tmp = dir.join(format!(
                ".{}-{}-{}.tmp",
                key.0,
                std::process::id(),
                NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
            ));
            match OpenOptions::new().write(true).create_new(true).open(&tmp) {
                Ok(f) => {
                    selected = Some((tmp, f));
                    break;
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e.into()),
            }
        }
        let (tmp, mut file) = selected
            .ok_or_else(|| Error::Invalid("unable to allocate cache temporary file".into()))?;
        struct Cleanup(PathBuf);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                let _ = fs::remove_file(&self.0);
            }
        }
        let _cleanup = Cleanup(tmp.clone());
        file.write_all(&header)?;
        file.write_all(&checksum)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        drop(file);
        // Hard-link publication is atomic, same-filesystem, and never overwrites
        // another writer. A concurrent winner is validated before being accepted.
        match fs::hard_link(&tmp, &path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                read_cache(&path, key)?;
            }
            Err(e) => return Err(e.into()),
        }
        Ok(CachedModel {
            model,
            hit: false,
            path: Some(path),
        })
    }
}
fn read_cache(path: &Path, key: &CacheKey) -> Result<Anny> {
    ensure(
        !fs::symlink_metadata(path)?.file_type().is_symlink(),
        "cache entry must not be a symlink",
    )?;
    let mut file = fs::File::open(path)?;
    let size = file.metadata()?.len();
    ensure(
        size >= (HEADER + 32) as u64 && size < MAX_CACHE_BYTES + (HEADER + 33) as u64,
        "invalid prepared cache size",
    )?;
    let mut header = [0u8; HEADER];
    file.read_exact(&mut header)?;
    ensure(
        &header[..8] == MAGIC && &header[8..72] == key.0.as_bytes(),
        "prepared cache version/key mismatch",
    )?;
    let len = u64::from_le_bytes(header[72..80].try_into().unwrap());
    ensure(
        len <= MAX_CACHE_BYTES && len + (HEADER + 32) as u64 == size,
        "prepared cache length mismatch",
    )?;
    let mut checksum = [0u8; 32];
    file.read_exact(&mut checksum)?;
    let mut payload = Vec::new();
    payload
        .try_reserve_exact(
            usize::try_from(len)
                .map_err(|_| Error::Invalid("cache does not fit this target".into()))?,
        )
        .map_err(|_| Error::Invalid("cache allocation failed".into()))?;
    file.take(len).read_to_end(&mut payload)?;
    ensure(
        payload.len() as u64 == len && Sha256::digest(&payload).as_slice() == checksum,
        "prepared cache checksum mismatch; remove the corrupt entry explicitly",
    )?;
    Anny::from_bytes(&payload, None)
}
/// Hash every regular file under assets, with sorted relative paths and explicit
/// length framing. Symlinks are rejected. Custom rig/weight files outside assets
/// participate too; touching bytes invalidates a cache even at unchanged mtimes.
pub fn asset_fingerprint(store: &AssetStore, config: &AnnyConfig) -> Result<String> {
    let root = store.root.canonicalize()?;
    fn walk(root: &Path, dir: &Path, out: &mut Vec<(String, PathBuf)>) -> Result<()> {
        for e in fs::read_dir(dir)? {
            let e = e?;
            let t = e.file_type()?;
            ensure(!t.is_symlink(), "symlink in cache asset dependencies")?;
            if t.is_dir() {
                walk(root, &e.path(), out)?;
            } else {
                ensure(t.is_file(), "non-regular cache asset dependency")?;
                let path = e.path();
                let rel = path
                    .strip_prefix(root)
                    .unwrap()
                    .to_str()
                    .ok_or_else(|| Error::Invalid("asset path must be UTF-8".into()))?
                    .replace('\\', "/");
                out.push((rel, path));
            }
        }
        Ok(())
    }
    let mut files = vec![];
    walk(&root, &root, &mut files)?;
    let rig = config.rig.resolve()?;
    let mut extra = vec![];
    if ![
        "anny",
        "makehuman",
        "cmu_mb",
        "game_engine",
        "mixamo",
        "soma",
    ]
    .contains(&rig.base_rig.as_str())
    {
        extra.push(("rig", root.join(&rig.base_rig)));
    }
    if let Some(w) = rig.weights_filename {
        extra.push(("weights", root.join(w)));
    }
    for (name, p) in extra {
        let p = p.canonicalize()?;
        if !p.starts_with(&root) {
            files.push((format!("@custom/{name}"), p));
        }
    }
    files.sort_by(|a, b| a.0.cmp(&b.0));
    let mut h = Sha256::new();
    for (name, p) in files {
        h.update((name.len() as u64).to_le_bytes());
        h.update(name.as_bytes());
        h.update(crate::import::hash_file(&p)?.as_bytes());
    }
    Ok(format!("{:x}", h.finalize()))
}
