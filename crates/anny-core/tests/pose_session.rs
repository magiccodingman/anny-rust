//! `Anny::pose_session` must be a pure fast path: for the same parameters it has to reproduce
//! `Anny::forward` output exactly, otherwise it is a second, subtly different evaluator.
//!
//! These tests therefore compare against `forward` rather than against hard-coded numbers — the
//! parity suite already pins `forward` to upstream, so pinning the session to `forward` inherits
//! that qualification instead of duplicating it.
mod common;
use anny_core::{config::*, math::*, model::identity_poses, *};
use serde_json::{json, Value};

/// A pose whose root is translated and whose second bone is rotated, so the session cannot pass by
/// simply reproducing identity. Poses reach both paths as JSON, so one builder serves f64 and f32.
fn posed_bones(bones: usize, root: (f64, f64, f64), joint: (f64, f64, f64)) -> Value {
    let mut pose = identity_poses(1, bones);
    write4(
        &rigid(
            &rotvec(&Vec3::new(root.0, root.1, root.2)),
            &Vec3::new(0.3, -0.2, 0.1),
        ),
        &mut pose.data[..16],
    );
    write4(
        &rigid(
            &rotvec(&Vec3::new(joint.0, joint.1, joint.2)),
            &Vec3::zeros(),
        ),
        &mut pose.data[16..32],
    );
    pose.nested_json()
}

fn posed(model: &Anny, root: (f64, f64, f64), joint: (f64, f64, f64)) -> Value {
    posed_bones(model.data.bone_count(), root, joint)
}

fn posed_f32(model: &AnnyF32, root: (f64, f64, f64), joint: (f64, f64, f64)) -> Value {
    posed_bones(model.data().bone_count(), root, joint)
}

fn max_difference(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "compared arrays differ in length");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// Compare a session result with a `forward` result key by key, allowing an error on both sides to
/// agree (for example `return_bone_ends` with a non-blender rig).
fn assert_same_output(session: Result<&ModelOutput>, direct: Result<ModelOutput>, case: &str) {
    match (session, direct) {
        (Ok(got), Ok(expected)) => {
            let mut keys: Vec<&String> = got.arrays.keys().collect();
            let mut expected_keys: Vec<&String> = expected.arrays.keys().collect();
            keys.sort();
            expected_keys.sort();
            assert_eq!(keys, expected_keys, "array keys differ for {case}");
            for key in keys {
                let g = got.get(key).unwrap();
                let e = expected.get(key).unwrap();
                assert_eq!(e.shape, g.shape, "{key} shape differs for {case}");
                let max = max_difference(&e.data, &g.data);
                // The session runs the same code on the same coefficients, so this is not a
                // tolerance-driven match: it should be exact, and a loose bound here would hide a
                // real divergence.
                assert!(max == 0.0, "{key} max difference {max:e} for {case}");
            }
        }
        (Err(_), Err(_)) => {}
        (Ok(_), Err(e)) => panic!("session succeeded but forward failed for {case}: {e}"),
        (Err(e), Ok(_)) => panic!("session failed but forward succeeded for {case}: {e}"),
    }
}

#[test]
fn session_update_reproduces_forward_across_every_convention() {
    let model = common::tiny();
    for mode in [
        PoseParameterization::LocalRef,
        PoseParameterization::LocalBone,
        PoseParameterization::LocalBoneWorld,
        PoseParameterization::World,
        PoseParameterization::WorldOrient,
    ] {
        let base = Parameters {
            pose_parameterization: Some(mode),
            ..Default::default()
        };
        let mut session = model.pose_session(&base).unwrap();
        for k in 0..5 {
            let pose = posed(
                &model,
                (0.02 * k as f64, -0.01, 0.03),
                (0.05 * k as f64, 0.1, -0.2),
            );
            let direct = model.forward(&Parameters {
                pose_parameters: pose.clone(),
                ..base.clone()
            });
            let via_session = session.update(&pose);
            assert_same_output(via_session, direct, &format!("mode {mode:?}, pose {k}"));
        }
    }
}

#[test]
fn session_keeps_the_rest_model_fixed_across_updates() {
    let model = common::tiny();
    let mut session = model.pose_session(&Parameters::default()).unwrap();
    let first = session
        .update(&posed(&model, (0., 0., 0.), (0., 0., 0.)))
        .unwrap();
    let rest = first.get("rest_vertices").unwrap().data.clone();
    let rest_poses = first.get("rest_bone_poses").unwrap().data.clone();
    for k in 1..10 {
        let out = session
            .update(&posed(
                &model,
                (0.01 * k as f64, 0.0, -0.02),
                (0.02 * k as f64, 0.0, 0.0),
            ))
            .unwrap();
        assert_eq!(out.get("rest_vertices").unwrap().data, rest);
        assert_eq!(out.get("rest_bone_poses").unwrap().data, rest_poses);
    }
}

#[test]
fn session_handles_batched_poses_and_batch_size_changes() {
    let model = common::tiny();
    let j = model.data.bone_count();
    let n = model.data.vertex_count();
    let mut two = identity_poses(2, j);
    for (slot, rotation) in [
        (0usize, Vec3::new(0.1, 0.0, 0.0)),
        (1usize, Vec3::new(0.0, 0.2, 0.0)),
    ] {
        write4(
            &rigid(&rotvec(&rotation), &Vec3::zeros()),
            &mut two.data[slot * j * 16..(slot + 1) * j * 16],
        );
    }
    let batched = two.nested_json();
    let mut session = model.pose_session(&Parameters::default()).unwrap();

    // Breadth first, then a two-pose batch, then back to one: changing the batch size exercises the
    // buffer-reuse path, which must fall back to allocation rather than hand back a stale shape.
    let got = session.update(&batched).unwrap();
    assert_eq!(got.get("vertices").unwrap().shape, vec![2, n, 3]);
    assert_eq!(got.get("bone_poses").unwrap().shape, vec![2, j, 4, 4]);
    let direct = model.forward(&Parameters {
        pose_parameters: batched.clone(),
        ..Default::default()
    });
    assert_same_output(Ok(got), direct, "batched pose");

    let single = posed(&model, (0.1, 0.1, 0.1), (0.1, 0.1, 0.1));
    let got = session.update(&single).unwrap();
    assert_eq!(got.get("vertices").unwrap().shape, vec![1, n, 3]);
    let direct = model.forward(&Parameters {
        pose_parameters: single,
        ..Default::default()
    });
    assert_same_output(Ok(got), direct, "batch size returned to one");
}

#[test]
fn a_rejected_pose_leaves_the_session_usable() {
    let model = common::tiny();
    let mut session = model.pose_session(&Parameters::default()).unwrap();
    let bad = json!({"not_a_bone": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]});
    assert!(
        session.update(&bad).is_err(),
        "an unknown bone name must be rejected"
    );

    // The failed update consumed the output it was handed, so this checks the session restored a
    // rest model instead of staying broken.
    let pose = posed(&model, (0.04, 0.0, 0.0), (0.0, 0.07, 0.0));
    let direct = model.forward(&Parameters {
        pose_parameters: pose.clone(),
        ..Default::default()
    });
    assert_same_output(session.update(&pose), direct, "after a rejected pose");
}

#[test]
fn session_honours_the_bone_end_setting() {
    let model = common::tiny();
    for return_bone_ends in [false, true] {
        let base = Parameters {
            return_bone_ends,
            ..Default::default()
        };
        let mut session = model.pose_session(&base).unwrap();
        let pose = posed(&model, (0.01, 0.02, 0.03), (0.05, 0.05, 0.05));
        let direct = model.forward(&Parameters {
            pose_parameters: pose.clone(),
            ..base.clone()
        });
        assert_same_output(
            session.update(&pose),
            direct,
            &format!("return_bone_ends = {return_bone_ends}"),
        );
    }
}

#[test]
fn session_coefficients_match_the_parameters_it_was_built_from() {
    let model = common::tiny();
    let p = Parameters {
        phenotype_kwargs: json!({"gender": 0.7}),
        facial_actions: json!({"jawOpen": 0.5}),
        local_changes_kwargs: json!({"test-pos": 1.0}),
        ..Default::default()
    };
    let session = model.pose_session(&p).unwrap();
    let expected = model.coefficients(&p).unwrap();
    assert_eq!(session.coefficients().shape, expected.shape);
    assert_eq!(session.coefficients().data, expected.data);
}

#[test]
fn typed_session_reproduces_typed_forward_and_is_bit_identical() {
    let model = common::tiny().to_f32().unwrap();
    for mode in [
        PoseParameterization::LocalRef,
        PoseParameterization::World,
        PoseParameterization::WorldOrient,
    ] {
        let base = Parameters {
            pose_parameterization: Some(mode),
            ..Default::default()
        };
        let mut session = model.pose_session(&base).unwrap();
        for k in 0..4 {
            let pose = posed_f32(&model, (0.02 * k as f64, -0.01, 0.03), (0.05, 0.1, -0.2));
            let expected = model
                .forward(&Parameters {
                    pose_parameters: pose.clone(),
                    ..base.clone()
                })
                .unwrap();
            let got = session.update(&pose).unwrap();
            for (key, value) in &expected.arrays {
                let other = got.get(key).unwrap();
                assert_eq!(value.shape, other.shape, "{key} shape ({mode:?})");
                assert_eq!(value.data, other.data, "{key} bits ({mode:?})");
            }
        }
    }
}

#[test]
fn typed_session_survives_a_rejected_pose() {
    let model = common::tiny().to_f32().unwrap();
    let mut session = model.pose_session(&Parameters::default()).unwrap();
    let bad = json!({"not_a_bone": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]});
    assert!(session.update(&bad).is_err());
    let pose = posed_f32(&model, (0.03, 0.0, 0.0), (0.0, 0.05, 0.0));
    let expected = model
        .forward(&Parameters {
            pose_parameters: pose.clone(),
            ..Default::default()
        })
        .unwrap();
    let got = session.update(&pose).unwrap();
    assert_eq!(
        expected.get("vertices").unwrap().data,
        got.get("vertices").unwrap().data
    );
}

/// Real data at the full model size on the typed path: this is the surface a game runtime uses, so
/// it is the f32 equivalence claim that actually has to hold.
#[ignore = "real-data typed session equivalence; not part of the fast suite"]
#[test]
fn typed_session_matches_typed_forward_on_the_real_model() -> Result<()> {
    let store = assets::AssetStore::new(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data"),
    );
    let model = store.build(&AnnyConfig::default())?.to_f32()?;
    let base = Parameters {
        phenotype_kwargs: json!({"gender": 0.62, "age": 0.37, "height": 0.55}),
        ..Default::default()
    };
    let mut session = model.pose_session(&base)?;
    let mut worst = 0.0f32;
    for k in 0..8 {
        let pose = posed_f32(
            &model,
            (0.01 * k as f64, -0.02, 0.03),
            (0.04 * k as f64, 0.03, -0.02),
        );
        let expected = model.forward(&Parameters {
            pose_parameters: pose.clone(),
            ..base.clone()
        })?;
        let got = session.update(&pose)?;
        for key in ["vertices", "bone_poses", "rest_vertices", "rest_bone_poses"] {
            let e = expected.get(key)?;
            let g = got.get(key)?;
            assert_eq!(e.shape, g.shape, "{key} shape");
            for (x, y) in e.data.iter().zip(&g.data) {
                worst = worst.max((x - y).abs());
            }
        }
    }
    eprintln!("typed session vs typed forward on the real model: max difference {worst:e}");
    assert_eq!(
        worst, 0.0,
        "the typed session must reproduce typed forward exactly"
    );
    Ok(())
}

/// Real data at the full model size: the synthetic fixture above is three vertices, so this checks
/// the equivalence claim on the committed model and reports the actual maximum difference.
#[ignore = "real-data session equivalence; not part of the fast suite"]
#[test]
fn session_matches_forward_on_the_real_model() -> Result<()> {
    let store = assets::AssetStore::new(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data"),
    );
    let model = store.build(&AnnyConfig::default())?;
    eprintln!(
        "real model: {} vertices, {} bones",
        model.data.vertex_count(),
        model.data.bone_count()
    );
    let base = Parameters {
        phenotype_kwargs: json!({"gender": 0.62, "age": 0.37, "height": 0.55}),
        ..Default::default()
    };
    let mut session = model.pose_session(&base)?;
    let mut worst = 0.0f64;
    let mut poses = 0usize;
    for k in 0..8 {
        let pose = posed(
            &model,
            (0.01 * k as f64, -0.02, 0.03),
            (0.04 * k as f64, 0.03, -0.02),
        );
        let expected = model.forward(&Parameters {
            pose_parameters: pose.clone(),
            ..base.clone()
        })?;
        let got = session.update(&pose)?;
        assert_eq!(
            expected.get("vertices")?.shape,
            got.get("vertices")?.shape,
            "vertex shape"
        );
        for key in ["vertices", "bone_poses", "rest_vertices", "rest_bone_poses"] {
            let e = expected.get(key)?;
            let g = got.get(key)?;
            assert_eq!(e.shape, g.shape, "{key} shape");
            worst = worst.max(max_difference(&e.data, &g.data));
        }
        poses += 1;
    }
    eprintln!("session vs forward on the real model: {poses} poses, max difference {worst:e}");
    assert_eq!(
        worst, 0.0,
        "the session must reproduce forward exactly on real data too"
    );
    Ok(())
}
