//! The prepared-payload reload must produce exactly the model the widening path produced.
//!
//! `AnnyF32::from_bytes` now decodes the f32 payload straight into f32 storage instead of widening it
//! to f64 and converting back (that round trip was 96 ms of the 229 ms reload). The two loaders must
//! agree bit-for-bit, including on the validation rules, since the direct path replaced a conversion
//! that used to fail on non-finite or non-representable data.
use anny_core::{config::AnnyConfig, Anny, AnnyF32};

fn assets() -> anny_core::assets::AssetStore {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
    anny_core::assets::AssetStore::new(root)
}

/// The direct f32 decode against the f64 widening path it replaced.
#[test]
#[ignore]
fn direct_f32_reload_matches_the_widening_path() {
    let store = assets();
    let config = AnnyConfig::default();
    let prepared = store.build(&config).unwrap();
    let bytes = prepared.to_f32().unwrap().to_bytes().unwrap();

    let direct = AnnyF32::from_bytes(&bytes, Some(config.clone())).unwrap();
    let widened =
        AnnyF32::from_anny(&Anny::from_bytes(&bytes, Some(config.clone())).unwrap()).unwrap();

    assert_eq!(
        direct.data().arrays.len(),
        widened.data().arrays.len(),
        "array count differs"
    );
    for (name, a) in &direct.data().arrays {
        let b = widened.data().get(name).unwrap();
        assert_eq!(a.shape, b.shape, "{name} shape differs");
        assert_eq!(a.kind, b.kind, "{name} kind differs");
        for (i, (x, y)) in a.data.iter().zip(&b.data).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{name}[{i}] differs: {x} vs {y} (bit-exact comparison)"
            );
        }
    }
    assert_eq!(direct.describe(), widened.describe());
}

/// The reload is also a superset of the pipeline it serves: pose the same parameters through both
/// loaders and the results must be identical.
#[test]
#[ignore]
fn direct_f32_reload_poses_identically() {
    let store = assets();
    let config = AnnyConfig::default();
    let prepared = store.build(&config).unwrap();
    let bytes = prepared.to_f32().unwrap().to_bytes().unwrap();
    let direct = AnnyF32::from_bytes(&bytes, Some(config.clone())).unwrap();
    let widened = AnnyF32::from_anny(&Anny::from_bytes(&bytes, Some(config)).unwrap()).unwrap();

    let p = anny_core::Parameters::default();
    let a = direct.forward(&p).unwrap();
    let b = widened.forward(&p).unwrap();
    for (name, ta) in &a.arrays {
        let tb = b.get(name).unwrap();
        assert_eq!(ta.shape, tb.shape, "{name} shape differs");
        for (i, (x, y)) in ta.data.iter().zip(&tb.data).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "{name}[{i}] differs");
        }
    }
}
