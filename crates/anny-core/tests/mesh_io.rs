mod common;
use anny_core::{mesh_io, scene::*, Parameters, Result};
#[test]
fn normal_geometry_formats_roundtrip() -> Result<()> {
    let mesh = SurfaceMesh {
        positions: vec![[0., 0., 0.], [1., 0., 0.], [0., 0., 1.]],
        triangles: vec![[0, 1, 2]],
        ..Default::default()
    };
    for fmt in ["obj", "ply", "stl"] {
        let data = mesh_io::to_bytes(&mesh, fmt)?;
        let m = mesh_io::from_bytes(&data, fmt)?;
        assert_eq!(m.positions, mesh.positions, "{fmt}");
        assert_eq!(m.triangles, mesh.triangles, "{fmt}");
    }
    Ok(())
}
#[test]
fn all_obj_objects_and_ply_extra_properties() -> Result<()> {
    let m = mesh_io::from_bytes(
        b"o first\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\no second\nv 2 0 0\nf 2 4 3\n",
        "obj",
    )?;
    assert_eq!(m.positions.len(), 4);
    assert_eq!(m.triangles.len(), 2);
    let ply=b"ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0 255\n1 0 0 128\n0 0 1 0\n3 0 1 2\n";
    let m = mesh_io::from_bytes(ply, "ply")?;
    assert_eq!(m.positions.len(), 3);
    assert_eq!(m.triangles, vec![[0, 1, 2]]);
    Ok(())
}
#[test]
fn glb_and_gltf_evaluate_skin_and_coordinate_system() -> Result<()> {
    let model = common::tiny();
    let params = Parameters::default();
    let output = model.forward(&params)?;
    for rigged in [false, true] {
        let mut scene = Scene::new();
        scene.add_character(
            &model,
            &params,
            &CharacterExport {
                rigged,
                translation: [2., 3., 4.],
                ..Default::default()
            },
        )?;
        for (bytes, format) in [(scene.to_glb()?, "glb"), (scene.to_gltf()?, "gltf")] {
            let mesh = mesh_io::from_bytes(&bytes, format)?;
            assert_eq!(mesh.positions.len(), 3);
            for (i, p) in mesh.positions.iter().enumerate() {
                for k in 0..3 {
                    let expected = output.get("vertices")?.data[i * 3 + k] + [2., 3., 4.][k];
                    assert!(
                        (p[k] - expected).abs() < 1e-5,
                        "{rigged} {format} {p:?} {expected}"
                    );
                }
            }
        }
    }
    Ok(())
}
#[test]
fn malformed_inputs_are_errors_not_panics() {
    for fmt in ["obj", "ply", "stl", "glb", "gltf"] {
        for bytes in [b"".as_slice(), b"glTF", b"junk"] {
            assert!(mesh_io::from_bytes(bytes, fmt).is_err());
        }
    }
    assert!(mesh_io::from_bytes(b"v 0 0 0\nf 0 1 3", "obj").is_err());
    assert!(mesh_io::from_bytes(b"{\"asset\":{\"version\":\"2.0\"},\"buffers\":[{\"uri\":\"../secret.bin\",\"byteLength\":4}]}","gltf").is_err());
}
