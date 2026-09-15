//! Native, deterministic Anny geometry. No Python or tensor-framework runtime.
//! See `docs/COMPATIBILITY.md` for the exact upstream revision and contracts.
#![forbid(unsafe_code)]
pub mod assets;
pub mod config;
pub mod differentiation;
pub mod distribution;
pub mod import;
pub mod inverter;
pub mod math;
pub mod mesh;
pub mod mesh_io;
pub mod model;
mod parallel;
pub mod scene;
pub mod smpl;
pub mod tensor;
pub mod tools;
pub mod torch_archive;

pub use config::{AnnyConfig, PoseParameterization, SkinningMethod};
pub use model::{Anny, ModelData, ModelOutput, Parameters, PoseSession};
pub use tensor::Tensor;
pub const UPSTREAM_REVISION: &str = "81ca83e202273b306205c1cc15f33734be31e48c";
pub const DATA_VERSION: usize = 11;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Invalid(String),
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Json(#[from] serde_json::Error),
    #[error("{0}")]
    Safetensors(#[from] safetensors::SafeTensorError),
}
pub type Result<T> = std::result::Result<T, Error>;
pub(crate) fn ensure(condition: bool, message: impl Into<String>) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(Error::Invalid(message.into()))
    }
}

pub mod transforms;

pub mod cache;
pub mod fitting;
pub mod precompute;

pub mod operations;

pub mod typed;
pub use typed::{AnnyF32, ModelOutputF32, PoseSessionF32, TensorF32};

pub mod motion;
pub mod numpy;

pub mod prior;

pub mod refinement;

pub mod gltf_asset;
