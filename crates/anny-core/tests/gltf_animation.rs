mod common;
use anny_core::{
    gltf_asset::*,
    scene::{CharacterExport, Scene},
    Parameters,
};
fn channel(interpolation: Interpolation) -> AnimationChannel {
    AnimationChannel {
        node: 0,
        path: AnimationPath::Translation,
        interpolation,
        times: vec![0., 2.],
        width: 3,
        values: vec![0., 0., 0., 2., 0., 0.],
    }
}
#[test]
fn step_linear_cubic_and_time_clamp() {
    let c = channel(Interpolation::Linear);
    assert_eq!(c.sample(-1.).unwrap(), vec![0., 0., 0.]);
    assert_eq!(c.sample(1.).unwrap(), vec![1., 0., 0.]);
    assert_eq!(c.sample(9.).unwrap(), vec![2., 0., 0.]);
    let c = channel(Interpolation::Step);
    assert_eq!(c.sample(1.99).unwrap()[0], 0.);
    assert_eq!(c.sample(2.).unwrap()[0], 2.);
    let mut c = channel(Interpolation::CubicSpline);
    c.values = vec![
        0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 2., 0., 0., 0., 0., 0.,
    ];
    assert_eq!(c.sample(1.).unwrap()[0], 1.25); // Tangent scaled by two-second interval.
    c.times[1] = 0.;
    assert!(c.sample(0.).is_err());
}
#[test]
fn quaternion_slerp_uses_shortest_arc_and_cubic_normalizes() {
    let mut c = AnimationChannel {
        path: AnimationPath::Rotation,
        width: 4,
        values: vec![0., 0., 0., 1., 0., 0., 1., 0.],
        ..channel(Interpolation::Linear)
    };
    let v = c.sample(1.).unwrap();
    assert!((v[2] - 0.5f64.sqrt()).abs() < 1e-12 && (v[3] - 0.5f64.sqrt()).abs() < 1e-12);
    c.values = vec![0., 0., 0., 1., 0., 0., 0., -1.];
    assert_eq!(c.sample(1.).unwrap(), vec![0., 0., 0., 1.]);
    c.interpolation = Interpolation::CubicSpline;
    c.values = vec![
        0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
        0.,
    ];
    let v = c.sample(1.).unwrap();
    assert!((v.iter().map(|x| x * x).sum::<f64>() - 1.).abs() < 1e-12);
}
#[test]
fn native_skeletal_animation_roundtrip_and_geometry_sampling() {
    let mut scene = Scene::new();
    scene
        .add_character(
            &common::tiny(),
            &Parameters::default(),
            &CharacterExport {
                rigged: true,
                ..Default::default()
            },
        )
        .unwrap();
    let mut asset = GltfAsset::from_scene(&scene).unwrap();
    let node = asset.document()["skins"][0]["joints"][0].as_u64().unwrap() as usize;
    let mut c = channel(Interpolation::Linear);
    c.node = node;
    c.values = vec![0., 0., 0., 0., 0., 2.];
    asset
        .add_animation(&AnimationClip {
            name: "rise".into(),
            channels: vec![c],
        })
        .unwrap();
    let start = asset.geometry_at(0, 0.).unwrap();
    let half = asset.geometry_at(0, 1.).unwrap();
    for (a, b) in start.positions.iter().zip(&half.positions) {
        assert!((b[2] - a[2] - 1.).abs() < 1e-5, "{a:?} {b:?}");
    }
    let again = GltfAsset::from_bytes(&asset.to_glb().unwrap()).unwrap();
    assert_eq!(
        again.document()["animations"],
        asset.document()["animations"]
    );
    assert_eq!(again.geometry_at(0, 1.).unwrap().positions, half.positions);
    assert_eq!(again.animation_clips().unwrap()[0].name, "rise");
}
#[test]
fn animation_edit_errors_do_not_mutate_document() {
    let mut scene = Scene::new();
    scene
        .add_character(
            &common::tiny(),
            &Parameters::default(),
            &CharacterExport {
                rigged: true,
                ..Default::default()
            },
        )
        .unwrap();
    let mut asset = GltfAsset::from_scene(&scene).unwrap();
    let old = asset.to_glb().unwrap();
    let c = AnimationChannel {
        node: 999,
        ..channel(Interpolation::Linear)
    };
    assert!(asset
        .add_animation(&AnimationClip {
            name: "bad".into(),
            channels: vec![c]
        })
        .is_err());
    assert_eq!(old, asset.to_glb().unwrap());
    assert!(asset.geometry_at(0, 0.).is_err());
}

fn animation_document() -> serde_json::Value {
    let mut scene = Scene::new();
    scene
        .add_character(
            &common::tiny(),
            &Parameters::default(),
            &CharacterExport {
                rigged: true,
                ..Default::default()
            },
        )
        .unwrap();
    let mut asset = GltfAsset::from_scene(&scene).unwrap();
    let node = asset.document()["skins"][0]["joints"][0].as_u64().unwrap() as usize;
    let c = AnimationChannel {
        node,
        ..channel(Interpolation::Linear)
    };
    asset
        .add_animation(&AnimationClip {
            name: "test".into(),
            channels: vec![c],
        })
        .unwrap();
    // Use the retained JSON with a single inline buffer so corrupt metadata can
    // be tested without invoking another glTF library or external filesystem I/O.
    let bytes = asset.to_glb().unwrap();
    let json_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let mut root: serde_json::Value = serde_json::from_slice(&bytes[20..20 + json_len]).unwrap();
    let start = 20 + json_len + 8;
    use base64::Engine;
    root["buffers"][0]["uri"] = serde_json::json!(format!(
        "data:application/octet-stream;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(&bytes[start..])
    ));
    root
}

#[test]
fn imported_animation_rejects_invalid_component_types_and_time_bounds() {
    use serde_json::json;
    let root = animation_document();
    let input = root["animations"][0]["samplers"][0]["input"]
        .as_u64()
        .unwrap() as usize;
    let output = root["animations"][0]["samplers"][0]["output"]
        .as_u64()
        .unwrap() as usize;
    assert!(GltfAsset::from_bytes(&serde_json::to_vec(&root).unwrap()).is_ok());
    for (id, key, value) in [
        (output, "componentType", json!(5125)),
        (output, "normalized", json!(true)),
        (input, "normalized", json!(true)),
        (input, "normalized", json!("false")),
        (input, "min", json!([-1.])),
        (input, "max", json!([3.])),
        (input, "min", json!([])),
    ] {
        let mut invalid = root.clone();
        invalid["accessors"][id][key] = value;
        assert!(
            GltfAsset::from_bytes(&serde_json::to_vec(&invalid).unwrap()).is_err(),
            "accepted {key}"
        );
    }
}

#[test]
fn normalized_integer_rotation_keys_remain_supported() {
    use base64::Engine;
    use serde_json::json;
    let mut root = animation_document();
    let output = root["animations"][0]["samplers"][0]["output"]
        .as_u64()
        .unwrap() as usize;
    let accessor = &mut root["accessors"][output];
    accessor["componentType"] = json!(5121);
    accessor["normalized"] = json!(true);
    accessor["type"] = json!("VEC4");
    let view = accessor["bufferView"].as_u64().unwrap() as usize;
    let start = root["bufferViews"][view]["byteOffset"]
        .as_u64()
        .unwrap_or(0) as usize;
    let uri = root["buffers"][0]["uri"].as_str().unwrap();
    let mut buffer = base64::engine::general_purpose::STANDARD
        .decode(uri.split_once(',').unwrap().1)
        .unwrap();
    buffer[start..start + 8].copy_from_slice(&[0, 0, 0, 255, 0, 0, 255, 0]);
    root["buffers"][0]["uri"] = json!(format!(
        "data:application/octet-stream;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(buffer)
    ));
    root["animations"][0]["channels"][0]["target"]["path"] = json!("rotation");
    let asset = GltfAsset::from_bytes(&serde_json::to_vec(&root).unwrap()).unwrap();
    let q = asset.animation_clips().unwrap()[0].channels[0]
        .sample(1.)
        .unwrap();
    assert!((q[2] - 0.5f64.sqrt()).abs() < 1e-12);
    root["accessors"][output]["normalized"] = json!(false);
    assert!(GltfAsset::from_bytes(&serde_json::to_vec(&root).unwrap()).is_err());
}

#[test]
fn decimal_time_bounds_roundtrip_in_accessor_precision() {
    let root = animation_document();
    let mut asset = GltfAsset::from_bytes(&serde_json::to_vec(&root).unwrap()).unwrap();
    let node = asset.document()["skins"][0]["joints"][0].as_u64().unwrap() as usize;
    let c = AnimationChannel {
        node,
        times: vec![0.1, 0.2],
        ..channel(Interpolation::Linear)
    };
    asset
        .add_animation(&AnimationClip {
            name: "fractional-time".into(),
            channels: vec![c],
        })
        .unwrap();
    let again = GltfAsset::from_bytes(&asset.to_glb().unwrap()).unwrap();
    assert_eq!(
        again.animation_clips().unwrap()[1].channels[0].times,
        vec![0.1f32 as f64, 0.2f32 as f64]
    );
}
