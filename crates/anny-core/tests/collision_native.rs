//! Collision (`SelfInterpenetrationModule`) coverage.
//!
//! This path had no test at all before, which is how a mask predicate inverted during an
//! optimization went unnoticed: the label masks decide which face pairs may collide, and flipping
//! that test silently changes the result from 940 to 27,420 reported partners on the committed
//! model. The real-data test below pins the output exactly; the synthetic test is a construction
//! smoke test.
use anny_core::{assets::AssetStore, config::*, tools::SelfInterpenetrationModule, *};

mod common;

#[test]
fn collision_module_constructs_and_runs_on_a_synthetic_model() -> Result<()> {
    let model = common::tiny();
    let module = SelfInterpenetrationModule::new(&model, false, false, false)?;
    let vertices = model
        .forward(&Parameters::default())?
        .get("vertices")?
        .clone();
    let out = module.forward(&vertices)?;
    // One partner slot per face: the fixture has one face, so shape [1, 1].
    assert_eq!(out.shape, vec![model.data.get("faces")?.shape[0], 1]);
    Ok(())
}

/// The committed model produces a specific set of colliding face pairs. Recorded from the
/// implementation before the label masks were interned, so this fails if the pair rule changes.
#[ignore = "real-data collision digest; not part of the fast suite"]
#[test]
fn real_model_collision_matches_the_recorded_digest() -> Result<()> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let store = AssetStore::new(root.join("data"));
    let model = store.build(&AnnyConfig::default())?;
    let vertices = model
        .forward(&Parameters::default())?
        .get("vertices")?
        .clone();
    let module = SelfInterpenetrationModule::new(&model, false, false, false)?;
    let out = module.forward(&vertices)?;

    let mut hits = 0usize;
    let mut checksum = 0.0f64;
    for (i, &x) in out.data.iter().enumerate() {
        if x >= 0. {
            hits += 1;
            checksum += (i as f64 + 1.0) * (x + 1.0);
        }
    }
    println!(
        "collision digest: hits={hits} checksum={checksum:.6} slots={}",
        out.data.len()
    );
    assert_eq!(out.data.len(), 27420);
    assert_eq!(
        hits, 940,
        "number of faces with a collision partner changed"
    );
    assert_eq!(checksum, 287201168586.0, "collision partners changed");
    Ok(())
}
