mod common;
use anny_core::{math::*, scene::*, Parameters, Result};
use serde_json::Value;
fn document(bytes: &[u8]) -> Value {
    assert_eq!(&bytes[..4], b"glTF");
    assert_eq!(
        u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize,
        bytes.len()
    );
    let n = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    serde_json::from_slice(&bytes[20..20 + n]).unwrap()
}
#[test]
fn multiple_meshes_rig_and_animation() -> Result<()> {
    let m = common::tiny();
    let mut s = Scene::new();
    s.add_character(
        &m,
        &Parameters::default(),
        &CharacterExport {
            rigged: true,
            ..Default::default()
        },
    )?;
    s.add_character(
        &m,
        &Parameters::default(),
        &CharacterExport {
            name: "Second".into(),
            translation: [2., 0., 0.],
            ..Default::default()
        },
    )?;
    let poses = s.objects[0].skin.as_ref().unwrap().pose.clone();
    let mut next = poses.clone();
    next[0][(0, 3)] += 0.2;
    next[1][(0, 3)] += 0.2;
    s.add_animation(Animation {
        name: "Move".into(),
        object: 0,
        times: vec![0., 1.],
        bone_poses: vec![poses, next],
    })?;
    let bytes = s.to_glb()?;
    let doc = document(&bytes);
    assert_eq!(doc["meshes"].as_array().unwrap().len(), 2);
    assert_eq!(doc["skins"].as_array().unwrap().len(), 1);
    assert_eq!(doc["skins"][0]["joints"].as_array().unwrap().len(), 2);
    assert_eq!(
        doc["animations"][0]["channels"].as_array().unwrap().len(),
        4
    );
    assert!(doc["nodes"][0]["rotation"].is_array());
    let bin = &bytes[20 + u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize + 8..];
    for v in doc["bufferViews"].as_array().unwrap() {
        let off = v["byteOffset"].as_u64().unwrap() as usize;
        let len = v["byteLength"].as_u64().unwrap() as usize;
        assert_eq!(off % 4, 0);
        assert!(off + len <= bin.len());
    }
    // Written only on explicit opt-in for the independent Khronos validator.
    if let Ok(path) = std::env::var("ANNY_GLTF_TEST_OUTPUT") {
        std::fs::write(path, bytes)?;
    }
    let embedded: Value = serde_json::from_slice(&s.to_gltf()?)?;
    assert!(embedded["buffers"][0]["uri"]
        .as_str()
        .unwrap()
        .starts_with("data:"));
    Ok(())
}
#[test]
fn uv_seams_are_split_without_changing_the_model() -> Result<()> {
    let mut m = common::tiny();
    m.data.put(
        "faces",
        anny_core::Tensor::indices(vec![2, 3], vec![0, 1, 2, 0, 2, 1]),
    );
    m.data.put(
        "texture_coordinates",
        anny_core::Tensor::new(vec![4, 2], vec![0., 0., 1., 0., 0., 1., 0.5, 0.5])?,
    );
    m.data.put(
        "face_texture_coordinate_indices",
        anny_core::Tensor::indices(vec![2, 3], vec![0, 1, 2, 3, 2, 1]),
    );
    let mut s = Scene::new();
    s.add_character(&m, &Parameters::default(), &CharacterExport::default())?;
    assert_eq!(s.objects[0].mesh.positions.len(), 4);
    assert_eq!(s.objects[0].mesh.source_vertex_indices, vec![0, 1, 2, 0]);
    assert_eq!(s.objects[0].mesh.texcoords[0], [0., 1.]);
    assert_eq!(m.data.vertex_count(), 3);
    Ok(())
}
#[test]
fn reject_dqs_rig_and_invalid_public_scene_data() -> Result<()> {
    let mut m = common::tiny();
    m.config.skinning_method = anny_core::SkinningMethod::Dqs;
    let mut s = Scene::new();
    assert!(s
        .add_character(
            &m,
            &Parameters::default(),
            &CharacterExport {
                rigged: true,
                ..Default::default()
            }
        )
        .is_err());
    s.add_character(&m, &Parameters::default(), &CharacterExport::default())?;
    s.objects[0].transform[(0, 0)] = 2.;
    assert!(s.to_glb().is_err());
    s.objects[0].transform = Mat4::identity();
    s.objects[0].mesh.triangles[0][0] = 100;
    assert!(s.to_glb().is_err());
    Ok(())
}
