mod common;
use anny_core::{
    gltf_asset::*,
    scene::{CharacterExport, Scene},
    Parameters,
};
fn asset() -> GltfAsset {
    let mut s = Scene::new();
    s.add_character(
        &common::tiny(),
        &Parameters::default(),
        &CharacterExport {
            rigged: true,
            ..Default::default()
        },
    )
    .unwrap();
    s.objects[0].mesh.texcoords = vec![[0., 0.], [1., 0.], [0., 1.]];
    GltfAsset::from_scene(&s).unwrap()
}
fn png() -> Vec<u8> {
    let mut bytes = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut bytes, 1, 1);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header()
            .unwrap()
            .write_image_data(&[200, 100, 50, 255])
            .unwrap();
    }
    bytes
}
#[test]
fn morph_channels_and_weight_animation_evaluate_after_roundtrip() {
    let mut a = asset();
    let initial = a.geometry().unwrap();
    let delta = MorphDeltas {
        positions: vec![[0., 0., 0.], [0.5, 0., 0.], [0., 0., 0.]],
        ..Default::default()
    };
    a.add_morph_target(0, "wide", std::slice::from_ref(&delta), 0.5)
        .unwrap();
    assert_eq!(
        a.document()["meshes"][0]["extras"]["targetNames"][0],
        "wide"
    );
    assert!((a.geometry().unwrap().positions[1][0] - initial.positions[1][0] - 0.25).abs() < 1e-6);
    let node = a.document()["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .position(|n| n.get("mesh").is_some())
        .unwrap();
    a.add_animation(&AnimationClip {
        name: "width".into(),
        channels: vec![AnimationChannel {
            node,
            path: AnimationPath::Weights,
            interpolation: Interpolation::Linear,
            width: 1,
            times: vec![0., 2.],
            values: vec![0., 1.],
        }],
    })
    .unwrap();
    let bytes = a.to_glb().unwrap();
    let b = GltfAsset::from_bytes(&bytes).unwrap();
    assert!(
        (b.geometry_at(0, 1.).unwrap().positions[1][0] - initial.positions[1][0] - 0.25).abs()
            < 1e-6
    );
    assert!(a.add_morph_target(0, "another", &[delta], 0.).is_err());
    assert_eq!(a.to_glb().unwrap(), bytes);
}
#[test]
fn material_texture_roundtrip_and_reference_validation() {
    let mut a = asset();
    let texture = a
        .add_texture(
            &png(),
            &TextureOptions {
                name: "test".into(),
                wrap_s: WrapMode::ClampToEdge,
                ..Default::default()
            },
        )
        .unwrap();
    let material = PbrMaterial {
        name: "skin".into(),
        roughness: 0.4,
        alpha_mode: AlphaMode::Mask,
        base_color_texture: Some(TextureReference {
            texture,
            tex_coord: 0,
        }),
        ..Default::default()
    };
    let id = a.set_material(0, 0, &material).unwrap();
    let b = GltfAsset::from_bytes(&a.to_glb().unwrap()).unwrap();
    assert_eq!(b.document()["materials"][id]["name"], "skin");
    assert_eq!(
        b.document()["materials"][id]["pbrMetallicRoughness"]["baseColorTexture"]["index"],
        texture
    );
    assert_eq!(b.document()["samplers"][0]["wrapS"], 33071);
    let before = a.to_glb().unwrap();
    let invalid = PbrMaterial {
        base_color_texture: Some(TextureReference {
            texture: 999,
            tex_coord: 0,
        }),
        ..Default::default()
    };
    assert!(a.set_material(0, 0, &invalid).is_err());
    assert_eq!(before, a.to_glb().unwrap());
}
#[test]
fn authoring_bounds_and_duplicate_names_are_errors() {
    let mut a = asset();
    let d = MorphDeltas {
        positions: vec![[0.; 3]; 3],
        ..Default::default()
    };
    assert!(a.add_morph_target(0, "bad-count", &[], 0.).is_err());
    a.add_morph_target(0, "okay", std::slice::from_ref(&d), 0.)
        .unwrap();
    assert!(a.add_morph_target(0, "okay", &[d], 0.).is_err());
    assert!(a
        .add_texture(b"not png or jpeg", &Default::default())
        .is_err());
    assert!(a
        .set_material(
            0,
            0,
            &PbrMaterial {
                roughness: f32::NAN,
                ..Default::default()
            }
        )
        .is_err());
}
