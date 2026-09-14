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
