mod common;
use anny_core::{
    gltf_asset::GltfAsset,
    scene::{CharacterExport, Scene},
    Parameters,
};
use base64::Engine;
use serde_json::{json, Value};
fn document() -> Value {
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
    serde_json::from_slice(&scene.to_gltf().unwrap()).unwrap()
}
#[test]
fn owned_document_retains_materials_images_skin_and_metadata() {
    let mut root = document();
    let png = b"\x89PNG\r\n\x1a\nopaque-test-payload";
    root["images"] = json!([{"uri":format!("data:image/png;base64,{}",base64::engine::general_purpose::STANDARD.encode(png)),"name":"texture"}]);
    root["textures"] = json!([{"source":0,"sampler":0}]);
    root["samplers"] = json!([{"wrapS":33071,"wrapT":10497}]);
    root["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"] = json!({"index":0});
    root["extras"] = json!({"author":"test","extensions":"user-metadata-not-glTF-extension"});
    let asset = GltfAsset::from_bytes(&serde_json::to_vec(&root).unwrap()).unwrap();
    assert!(asset.document()["images"][0].get("uri").is_none());
    let bytes = asset.to_glb().unwrap();
    let again = GltfAsset::from_bytes(&bytes).unwrap();
    assert_eq!(again.document()["materials"], root["materials"]);
    assert_eq!(again.document()["skins"], root["skins"]);
    assert_eq!(again.document()["textures"], root["textures"]);
    assert_eq!(again.document()["samplers"], root["samplers"]);
    assert_eq!(again.document()["extras"], root["extras"]);
    assert_eq!(
        asset.geometry().unwrap().positions,
        again.geometry().unwrap().positions
    );
    assert!(bytes.windows(png.len()).any(|window| window == png));
}
#[test]
fn rejects_external_images_without_base_path_and_unsupported_extensions() {
    for image in [
        json!({"uri":"../not-allowed.png"}),
        json!({"uri":"image.png"}),
        json!({"uri":"https://example.com/image.png"}),
        json!({"uri":"data:image/png;base64,AAAA"}),
    ] {
        let mut root = document();
        root["images"] = json!([image]);
        assert!(GltfAsset::from_bytes(&serde_json::to_vec(&root).unwrap()).is_err());
    }
    let mut root = document();
    root["extensionsUsed"] = json!(["KHR_draco_mesh_compression"]);
    assert!(GltfAsset::from_bytes(&serde_json::to_vec(&root).unwrap()).is_err());
}
#[test]
fn geometry_import_retains_uv_zero() {
    let mut scene = Scene::new();
    scene
        .add_character(
            &common::tiny(),
            &Parameters::default(),
            &CharacterExport::default(),
        )
        .unwrap();
    scene.objects[0].mesh.texcoords = vec![[0., 0.], [1., 0.], [0., 1.]];
    let asset = GltfAsset::from_scene(&scene).unwrap();
    assert_eq!(
        asset.geometry().unwrap().texcoords,
        scene.objects[0].mesh.texcoords
    );
}
