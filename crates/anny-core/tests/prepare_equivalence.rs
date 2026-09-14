//! The prepared payload is assembled by a parallel target loader and a parallel gather over the
//! kept vertices, so what has to be pinned is the *content* it produces, not the bytes of the file:
//! the safetensors header's `__metadata__` is a `HashMap` whose JSON key order follows the process
//! hash seed, which makes whole-file digests differ between runs of the same binary.
//!
//! This digest was recorded from the sequential loader before the loads were parallelised, so it
//! fails if any of them changes a single element, or reorders the shapes, labels or masks.
use anny_core::{assets::AssetStore, config::AnnyConfig, *};
use sha2::{Digest, Sha256};

/// sha256 over the tensor region of `prepare default`, i.e. everything after the JSON header.
const DEFAULT_TENSOR_DIGEST: &str =
    "582aec10eda939b71619ca10d4d84d576d206841f336a34240030a743666011b";

#[test]
#[ignore = "real committed asset comparison; run cargo test -p anny-core --release --test prepare_equivalence -- --ignored"]
fn default_prepared_payload_matches_the_recorded_digest() -> Result<()> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let store = AssetStore::new(root.join("data"));
    let bytes = store.build(&AnnyConfig::default())?.to_f32()?.to_bytes()?;
    let header = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let mut hasher = Sha256::new();
    hasher.update(&bytes[8 + header..]);
    let digest: String = hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(
        digest, DEFAULT_TENSOR_DIGEST,
        "prepared payload tensors changed"
    );
    Ok(())
}
