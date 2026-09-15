use anny_core::{config::*, tensor::Kind, *};
use safetensors::{Dtype, SafeTensors};
use serde_json::json;
mod common;

fn difference(a: &[f64], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x - y as f64).abs())
        .fold(0., f64::max)
}
#[test]
fn single_precision_modes_skinning_and_serialization() {
    for skinning in [SkinningMethod::Lbs, SkinningMethod::Dqs] {
        let mut m = common::tiny();
        m.config.skinning_method = skinning;
        let typed = m.to_f32().unwrap();
        for mode in [
            PoseParameterization::LocalRef,
            PoseParameterization::LocalBone,
            PoseParameterization::LocalBoneWorld,
            PoseParameterization::World,
            PoseParameterization::WorldOrient,
        ] {
            let p = Parameters {
                pose_parameterization: Some(mode),
                local_changes_kwargs: json!({"test-pos":[0.3,-0.2]}),
                facial_actions: json!({"jawOpen":0.1}),
                ..Default::default()
            };
            let a = m.forward(&p).unwrap();
            let b = typed.forward(&p).unwrap();
            for (name, t) in &a.arrays {
                assert_eq!(t.shape, b.get(name).unwrap().shape);
                assert!(
                    difference(&t.data, &b.get(name).unwrap().data) < 2e-6,
                    "{name} {mode:?}"
                );
            }
            assert_eq!(b.get("vertices").unwrap().shape[0], 2);
            let bytes = b.to_bytes().unwrap();
            assert_eq!(
                SafeTensors::deserialize(&bytes)
                    .unwrap()
                    .tensor("vertices")
                    .unwrap()
                    .dtype(),
                Dtype::F32
            );
            let restored = AnnyF32::from_bytes(&typed.to_bytes().unwrap(), None).unwrap();
            assert_eq!(
                restored.forward(&p).unwrap().get("vertices").unwrap().data,
                b.get("vertices").unwrap().data
            );
            let pose = typed.pose_parameters(&b, mode).unwrap();
            assert_eq!(pose.shape, vec![2, 2, 4, 4]);
        }
    }
}
#[test]
fn typed_rejects_overflow_and_rounded_indices() {
    assert!(TensorF32::from_reference(&Tensor::new(vec![1], vec![f64::MAX]).unwrap()).is_err());
    let t = Tensor {
        shape: vec![1],
        data: vec![16_777_217.],
        kind: Kind::Index,
    };
    assert!(TensorF32::from_reference(&t).is_err());
    assert!(TensorF32::indices(vec![1], vec![16_777_217]).is_err());
    let m = common::tiny().to_f32().unwrap();
    assert!(m
        .forward(&Parameters {
            phenotype_kwargs: json!({"height":1e39}),
            ..Default::default()
        })
        .is_err());
    assert!(m
        .forward(&Parameters {
            phenotype_kwargs: json!({"not-a-control":1.}),
            ..Default::default()
        })
        .is_err());
}
#[test]
fn f32_evaluation_is_not_output_casting() {
    // Cancellation distinguishes arithmetic performed on f32 buffers from f64
    // arithmetic converted at the very end.
    let mut m = common::tiny();
    m.data.arrays.get_mut("template_vertices").unwrap().data[0] = 16_777_216.;
    m.data.arrays.get_mut("blendshapes").unwrap().data[0] = 1.;
    let a = m.forward(&Parameters::default()).unwrap();
    let b = m.to_f32().unwrap().forward(&Parameters::default()).unwrap();
    assert_eq!(a.get("rest_vertices").unwrap().data[0], 16_777_217.);
    assert_eq!(b.get("rest_vertices").unwrap().data[0], 16_777_216.);
    // A second accumulation is rounded at each f32 step, not once after summation.
    m.data.arrays.get_mut("blendshapes").unwrap().data[9] = 1.;
    let p = Parameters {
        facial_actions: json!({"jawOpen":1.}),
        ..Default::default()
    };
    let a = m.forward(&p).unwrap();
    let b = m.to_f32().unwrap().forward(&p).unwrap();
    assert_ne!(
        a.get("rest_vertices").unwrap().data[0] as f32,
        b.get("rest_vertices").unwrap().data[0]
    );
}
#[test]
#[ignore = "real asset qualification"]
fn real_configurations_f32_f64() {
    let assets = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
    let cases = include_str!("typed_cases.json");
    let cases: serde_json::Map<String, serde_json::Value> = serde_json::from_str(cases).unwrap();
    for (name, cfg) in cases {
        let c: AnnyConfig = serde_json::from_value(cfg).unwrap();
        let m = anny_core::assets::AssetStore::new(&assets)
            .build(&c)
            .unwrap();
        let m32 = m.to_f32().unwrap();
        let ph: serde_json::Map<_, _> = m
            .phenotype_labels
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), json!(0.25 + 0.5 * ((i * 7) % 11) as f64 / 10.)))
            .collect();
        let local: serde_json::Map<_, _> = m
            .local_change_labels
            .iter()
            .step_by(29)
            .enumerate()
            .map(|(i, n)| (n.clone(), json!(if i % 2 == 0 { 0.2 } else { -0.3 })))
            .collect();
        let mut p = Parameters {
            phenotype_kwargs: ph.into(),
            local_changes_kwargs: local.into(),
            ..Default::default()
        };
        let mut pose = anny_core::model::identity_poses(1, m.data.bone_count());
        for (i, row) in pose.data.chunks_exact_mut(16).enumerate() {
            let a = 0.12 * (i as f64 * 0.37).sin();
            row[0] = a.cos();
            row[1] = -a.sin();
            row[4] = a.sin();
            row[5] = a.cos();
        }
        pose.data[3] = 0.03;
        pose.data[7] = -0.02;
        pose.data[11] = 0.08;
        p.pose_parameters = pose.nested_json();
        let a = m.forward(&p).unwrap();
        let b = m32.forward(&p).unwrap();
        let mut largest: f64 = 0.;
        for (key, t) in &a.arrays {
            let err = difference(&t.data, &b.get(key).unwrap().data);
            largest = largest.max(err);
            assert!(err < 2e-4, "{name} {key} f32 error {err}");
        }
        println!("{name}: max array error {largest:.9e}");
    }
}
