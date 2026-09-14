use anny_core::{config::*, model::ModelMetadata, *};
pub fn tiny() -> Anny {
    let mut d = ModelData {
        metadata: ModelMetadata {
            bone_labels: vec!["root".into(), "joint".into()],
            bone_parents: vec![-1, 0],
            blendshape_labels: vec![
                "universal:fixture".into(),
                "facial_action:jawOpen".into(),
                "local_change:test-pos".into(),
                "local_change:test-neg".into(),
            ],
        },
        ..Default::default()
    };
    d.put(
        "template_vertices",
        Tensor::new(vec![3, 3], vec![0., 0., 0., 1., 0., 0., 0., 0., 1.]).unwrap(),
    );
    d.put("faces", Tensor::indices(vec![1, 3], vec![0, 1, 2]));
    d.put(
        "base_mesh_vertex_indices",
        Tensor::indices(vec![3], vec![0, 1, 2]),
    );
    let mut shapes = Tensor::zeros(vec![4, 3, 3]);
    shapes.data[9 + 7] = 0.2;
    shapes.data[18 + 3] = 0.5;
    shapes.data[27 + 3] = -0.3;
    d.put("blendshapes", shapes);
    d.put(
        "stacked_phenotype_blend_shapes_mask",
        Tensor::zeros(vec![1, 26]),
    );
    d.put(
        "template_bone_heads",
        Tensor::new(vec![2, 3], vec![0., 0., 0., 0., 0., 1.]).unwrap(),
    );
    d.put(
        "template_bone_tails",
        Tensor::new(vec![2, 3], vec![0., 1., 0., 0., 1., 1.]).unwrap(),
    );
    d.put("bone_heads_blendshapes", Tensor::zeros(vec![4, 2, 3]));
    d.put("bone_tails_blendshapes", Tensor::zeros(vec![4, 2, 3]));
    d.put(
        "bone_rolls_rotmat",
        Tensor::new(
            vec![1, 2, 3, 3],
            [1., 0., 0., 0., -1., 0., 0., 0., -1.].repeat(2),
        )
        .unwrap(),
    );
    d.put(
        "vertex_bone_weights",
        Tensor::new(vec![3, 2], vec![1., 0., 1., 0., 0., 1.]).unwrap(),
    );
    d.put(
        "vertex_bone_indices",
        Tensor::indices(vec![3, 2], vec![0, 1, 0, 1, 0, 1]),
    );
    Anny::from_model_data(
        d,
        AnnyConfig {
            rig: RigSpec::Name("makehuman".into()),
            facial_actions: Selection::Preset("all".into()),
            local_changes: Selection::Preset("all".into()),
            ..Default::default()
        },
    )
    .unwrap()
}
