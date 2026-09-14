mod common;
use anny_core::{fitting::*, scene::*, *};
#[test]
fn fitting_requires_explicit_index_semantics_and_supports_surfaces() -> Result<()> {
    let m = common::tiny();
    let p = Parameters::default();
    let mut scene = Scene::new();
    scene.add_character(&m, &p, &CharacterExport::default())?;
    let mesh = &scene.objects[0].mesh;
    let mut options = MeshFitOptions::default();
    options.initial.optimize_phenotypes = false;
    options.initial.max_n_iters = Some(0);
    options.outer_iterations = 1;
    options.inner_iterations = 1;
    let result = fit_mesh(&m, mesh, Correspondence::Index, &options)?;
    assert!(result.fit.mean_vertex_error[0] < 1e-9);
    let result = fit_mesh(&m, mesh, Correspondence::ClosestSurface, &options)?;
    assert!(result.distances.last().unwrap().is_finite());
    let mut broken = mesh.clone();
    broken.source_vertex_indices = vec![0, 0, 2];
    assert!(corresponding_vertices(&broken, 3).is_err());
    let loaded = mesh_io::from_bytes(&scene.to_glb()?, "glb")?;
    assert_eq!(loaded.source_vertex_indices, vec![0, 1, 2]);
    assert_eq!(corresponding_vertices(&loaded, 3)?.shape, [1, 3, 3]);
    Ok(())
}
