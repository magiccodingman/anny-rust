mod common;
use anny_core::{cache::*, *};
struct Temp(std::path::PathBuf);
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
#[test]
fn cache_hits_checksums_and_content_keys() -> Result<()> {
    let t = Temp(std::env::temp_dir().join(format!("anny-cache-test-{}", std::process::id())));
    std::fs::create_dir_all(&t.0)?;
    let model = common::tiny();
    let key = CacheKey::for_config(&model.config, &"a".repeat(64))?;
    let cache = ModelCache::directory(&t.0);
    let miss = cache.load_or_build_with(&key, || Ok(model.clone()))?;
    assert!(!miss.hit);
    assert!(
        cache
            .load_or_build_with(&key, || panic!("cache hit must not build"))?
            .hit
    );
    let path = miss.path.unwrap();
    let mut bytes = std::fs::read(&path)?;
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    std::fs::write(path, bytes)?;
    assert!(cache
        .load_or_build_with(&key, || panic!("do not silently replace corrupt cache"))
        .is_err());
    let assets = t.0.join("assets");
    std::fs::create_dir(&assets)?;
    std::fs::write(assets.join("one"), b"first")?;
    let store = assets::AssetStore::new(assets.clone());
    let a = asset_fingerprint(&store, &model.config)?;
    std::fs::write(assets.join("one"), b"other")?;
    let b = asset_fingerprint(&store, &model.config)?;
    assert_ne!(a, b);
    assert_ne!(
        CacheKey::for_config(&model.config, &a)?,
        CacheKey::for_config(&model.config, &b)?
    );
    assert!(ModelCache::directory(assets.join("bad-cache"))
        .load_or_build(&store, &model.config)
        .is_err());
    Ok(())
}
