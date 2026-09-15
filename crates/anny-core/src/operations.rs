//! Shared, serializable secondary API used identically by Rust, C/C#, and WASM.
//! This control-plane API favors explicit portable data over borrowed foreign
//! tensor objects. Forward evaluation retains its separate native array API.
use crate::{
    distribution::{SampleOptions, SimpleShapeDistribution},
    fitting::{Correspondence, MeshFitOptions},
    inverter::{FitOptions, InverterOptions},
    scene::SurfaceMesh,
    tools::{Anthropometry, KeypointsRegressor, SelfInterpenetrationModule},
    Anny, Parameters, PoseParameterization, Result, Tensor,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "kebab-case", deny_unknown_fields)]
pub enum Request {
    Jvp {
        #[serde(default)]
        parameters: Parameters,
        direction: crate::differentiation::ParameterDirection,
    },
    Vjp {
        #[serde(default)]
        parameters: Parameters,
        selection: crate::differentiation::ParameterSelection,
        cotangents: std::collections::BTreeMap<String, Tensor>,
    },
    Refine {
        target: Tensor,
        #[serde(default)]
        parameters: Parameters,
        #[serde(default)]
        options: Box<crate::refinement::RefinementOptions>,
        #[serde(default)]
        distribution: Option<SimpleShapeDistribution>,
    },
    PriorGradient {
        distribution: SimpleShapeDistribution,
        phenotypes: std::collections::BTreeMap<String, f64>,
    },
    MotionResample {
        clip: crate::motion::PoseClip,
        fps: f64,
    },
    AlignLandmarks {
        landmarks: crate::fitting::Landmarks,
    },
    Measure {
        #[serde(default)]
        parameters: Parameters,
    },
    Keypoints {
        #[serde(default)]
        parameters: Parameters,
        regressor: KeypointsRegressor,
    },
    PoseConvert {
        #[serde(default)]
        parameters: Parameters,
        mode: PoseParameterization,
    },
    Sample {
        distribution: SimpleShapeDistribution,
        #[serde(default)]
        options: SampleOptions,
    },
    PriorLoss {
        distribution: SimpleShapeDistribution,
        phenotypes: std::collections::BTreeMap<String, f64>,
    },
    Fit {
        target: Tensor,
        #[serde(default)]
        setup: InverterOptions,
        #[serde(default)]
        options: FitOptions,
    },
    FitMesh {
        mesh: SurfaceMesh,
        correspondence: Correspondence,
        #[serde(default)]
        options: Box<MeshFitOptions>,
    },
    Collision {
        #[serde(default)]
        parameters: Parameters,
        #[serde(default = "yes")]
        group_toes: bool,
        #[serde(default = "yes")]
        group_eyes: bool,
        #[serde(default = "yes")]
        group_tongue: bool,
    },
}
fn yes() -> bool {
    true
}
pub fn execute(model: &Anny, request: &Request) -> Result<Value> {
    match request {
        Request::Jvp {
            parameters,
            direction,
        } => Ok(crate::differentiation::jvp(model, parameters, direction)?.to_json()),
        Request::Vjp {
            parameters,
            selection,
            cotangents,
        } => Ok(serde_json::to_value(crate::differentiation::vjp(
            model, parameters, selection, cotangents,
        )?)?),
        Request::Refine {
            target,
            parameters,
            options,
            distribution,
        } => Ok(crate::refinement::refine(
            model,
            target,
            parameters,
            options,
            distribution.as_ref(),
        )?
        .to_json()),
        Request::PriorGradient {
            distribution,
            phenotypes,
        } => {
            crate::ensure(
                distribution.phenotype_labels == model.phenotype_labels,
                "distribution phenotype labels do not match model",
            )?;
            let (loss, gradient) = crate::prior::shape_prior_gradient(distribution, phenotypes)?;
            Ok(json!({"loss":loss,"gradient":gradient}))
        }
        Request::MotionResample { clip, fps } => {
            Ok(serde_json::to_value(clip.resample(model, *fps)?)?)
        }
        Request::AlignLandmarks { landmarks } => Ok(serde_json::to_value(landmarks.align()?)?),
        Request::Measure { parameters } => {
            let result = model.forward(parameters)?;
            Ok(serde_json::to_value(
                Anthropometry::new(model)?.measure(result.get("rest_vertices")?)?,
            )?)
        }
        Request::Keypoints {
            parameters,
            regressor,
        } => {
            let result = model.forward(parameters)?;
            Ok(
                json!({"labels":regressor.labels,"keypoints":regressor.regress(result.get("vertices")?)?}),
            )
        }
        Request::PoseConvert { parameters, mode } => {
            let result = model.forward(parameters)?;
            Ok(
                json!({"pose_parameterization":mode,"pose_parameters":model.pose_parameters(&result,*mode)?.nested_json()}),
            )
        }
        Request::Sample {
            distribution,
            options,
        } => {
            crate::ensure(
                options.count <= 1_000_000,
                "control-plane sampling limit is 1000000 characters",
            )?;
            crate::ensure(
                distribution.phenotype_labels == model.phenotype_labels,
                "distribution phenotype labels do not match model",
            )?;
            Ok(serde_json::to_value(distribution.sample(options)?)?)
        }
        Request::PriorLoss {
            distribution,
            phenotypes,
        } => Ok(json!({"loss":distribution.prior_loss(phenotypes)?})),
        Request::Fit {
            target,
            setup,
            options,
        } => Ok(crate::inverter::AnnyInverter::new(model, setup.clone())?
            .fit(target, options)?
            .to_json()),
        Request::FitMesh {
            mesh,
            correspondence,
            options,
        } => Ok(crate::fitting::fit_mesh(model, mesh, *correspondence, options)?.to_json()),
        Request::Collision {
            parameters,
            group_toes,
            group_eyes,
            group_tongue,
        } => {
            let result = model.forward(parameters)?;
            let collision =
                SelfInterpenetrationModule::new(model, *group_toes, *group_eyes, *group_tongue)?;
            Ok(
                json!({"face_partners":collision.forward(result.get("vertices")?)?,"ordering":"deterministic CPU face IDs"}),
            )
        }
    }
}
pub fn execute_json(model: &Anny, request: &str) -> Result<String> {
    let request = serde_json::from_str(request)?;
    Ok(serde_json::to_string(&execute(model, &request)?)?)
}
