//! Browser wrapper. Supply prepared ModelData bytes; filesystem/asset construction
//! and networking are deliberately left to the host, outside the browser runtime.
use anny_core::{Anny, AnnyConfig, ModelOutput, Parameters, PoseSession, Tensor};
use std::sync::Arc;
use wasm_bindgen::prelude::*;

pub mod gpu;
fn js_error(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}
#[wasm_bindgen]
pub struct AnnyModel {
    model: Arc<Anny>,
}
#[wasm_bindgen]
pub struct AnnyResult {
    output: ModelOutput,
}
#[wasm_bindgen]
impl AnnyModel {
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: &[u8], config_json: Option<String>) -> Result<AnnyModel, JsValue> {
        let config = config_json
            .as_deref()
            .map(serde_json::from_str::<AnnyConfig>)
            .transpose()
            .map_err(js_error)?;
        Ok(Self {
            model: Arc::new(Anny::from_bytes(bytes, config).map_err(js_error)?),
        })
    }
    /// Owned bytes suitable for a Blob download; no filesystem/server dependency.
    pub fn export_glb(
        &self,
        parameters_json: Option<String>,
        options_json: Option<String>,
    ) -> Result<js_sys::Uint8Array, JsValue> {
        let p: Parameters = parameters_json
            .as_deref()
            .map(serde_json::from_str)
            .transpose()
            .map_err(js_error)?
            .unwrap_or_default();
        let o: anny_core::scene::CharacterExport = options_json
            .as_deref()
            .map(serde_json::from_str)
            .transpose()
            .map_err(js_error)?
            .unwrap_or_default();
        let mut scene = anny_core::scene::Scene::new();
        scene.add_character(&self.model, &p, &o).map_err(js_error)?;
        Ok(js_sys::Uint8Array::from(
            scene.to_glb().map_err(js_error)?.as_slice(),
        ))
    }
    /// Portable secondary API; request/response schemas match the native C API.
    pub fn query(&self, request_json: &str) -> Result<String, JsValue> {
        anny_core::operations::execute_json(&self.model, request_json).map_err(js_error)
    }
    pub fn transform(&self, operations_json: &str) -> Result<AnnyModel, JsValue> {
        let operations =
            serde_json::from_str::<Vec<anny_core::transforms::Transform>>(operations_json)
                .map_err(js_error)?;
        Ok(AnnyModel {
            model: Arc::new(
                anny_core::transforms::apply_pipeline(&self.model, &operations)
                    .map_err(js_error)?,
            ),
        })
    }
    pub fn prepared_bytes(&self) -> Result<js_sys::Uint8Array, JsValue> {
        let bytes = self
            .model
            .data
            .archive(Some(&self.model.config))
            .and_then(|a| a.to_bytes())
            .map_err(js_error)?;
        Ok(js_sys::Uint8Array::from(bytes.as_slice()))
    }
    pub fn transfer_pose(
        &self,
        target: &AnnyModel,
        parameters_json: &str,
        mode: &str,
    ) -> Result<String, JsValue> {
        let params = serde_json::from_str(parameters_json).map_err(js_error)?;
        let mode: anny_core::PoseParameterization =
            serde_json::from_value(serde_json::json!(mode)).map_err(js_error)?;
        let t =
            anny_core::tools::transfer_pose_parameters(&self.model, &target.model, &params, mode)
                .map_err(js_error)?;
        Ok(
            serde_json::json!({"pose_parameterization":mode,"pose_parameters":t.nested_json()})
                .to_string(),
        )
    }
    pub fn describe(&self) -> String {
        self.model.describe().to_string()
    }
    pub fn evaluate(&self, parameters_json: Option<String>) -> Result<AnnyResult, JsValue> {
        let parameters: Parameters = parameters_json
            .as_deref()
            .map(serde_json::from_str)
            .transpose()
            .map_err(js_error)?
            .unwrap_or_default();
        Ok(AnnyResult {
            output: self.model.forward(&parameters).map_err(js_error)?,
        })
    }
    pub fn tensor(&self, name: &str) -> Result<js_sys::Float64Array, JsValue> {
        Ok(js_sys::Float64Array::from(
            self.model.data.get(name).map_err(js_error)?.data.as_slice(),
        ))
    }
    pub fn indices(&self, name: &str) -> Result<js_sys::Uint32Array, JsValue> {
        indices(self.model.data.get(name).map_err(js_error)?)
    }
    pub fn shape(&self, name: &str) -> Result<js_sys::Uint32Array, JsValue> {
        shape(self.model.data.get(name).map_err(js_error)?)
    }
}
#[wasm_bindgen]
impl AnnyResult {
    /// Returns an owned JS typed-array copy, safe across WASM memory growth/free.
    pub fn tensor(&self, name: &str) -> Result<js_sys::Float64Array, JsValue> {
        Ok(js_sys::Float64Array::from(
            self.output.get(name).map_err(js_error)?.data.as_slice(),
        ))
    }
    pub fn shape(&self, name: &str) -> Result<js_sys::Uint32Array, JsValue> {
        shape(self.output.get(name).map_err(js_error)?)
    }
}
fn shape(t: &Tensor) -> Result<js_sys::Uint32Array, JsValue> {
    let v = t
        .shape
        .iter()
        .map(|&n| u32::try_from(n).map_err(js_error))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(js_sys::Uint32Array::from(v.as_slice()))
}
fn indices(t: &Tensor) -> Result<js_sys::Uint32Array, JsValue> {
    let mut values = Vec::with_capacity(t.data.len());
    for &x in &t.data {
        if !x.is_finite() || x < 0. || x > u32::MAX as f64 || x.fract() != 0. {
            return Err(js_error("tensor cannot be represented as u32 indices"));
        }
        values.push(x as u32);
    }
    Ok(js_sys::Uint32Array::from(values.as_slice()))
}

/// A reusable pose session: the coefficients and the rest model are evaluated once, so `update` pays
/// only for the pose. It holds its own reference to the model, so the model object may be dropped.
#[wasm_bindgen]
pub struct AnnySession {
    // SAFETY INVARIANT: fields drop in declaration order. The widened borrowed session must be
    // destroyed before the Arc owner releases the model allocation.
    session: PoseSession<'static>,
    _model: Arc<Anny>,
}
#[wasm_bindgen]
impl AnnySession {
    /// Re-pose the session. Arrays read before this call stay valid because they are copies.
    pub fn update(&mut self, pose_json: Option<String>) -> Result<(), JsValue> {
        let pose: serde_json::Value = pose_json
            .as_deref()
            .map(serde_json::from_str)
            .transpose()
            .map_err(js_error)?
            .unwrap_or_default();
        self.session.update(&pose).map_err(js_error)?;
        Ok(())
    }
    /// Owned copy of one array of the current pose (the rest model before any `update`).
    pub fn tensor(&self, name: &str) -> Result<js_sys::Float64Array, JsValue> {
        Ok(js_sys::Float64Array::from(
            self.session
                .output()
                .get(name)
                .map_err(js_error)?
                .data
                .as_slice(),
        ))
    }
    pub fn shape(&self, name: &str) -> Result<js_sys::Uint32Array, JsValue> {
        shape(self.session.output().get(name).map_err(js_error)?)
    }
    /// Owned copy of the coefficients this session was created with.
    pub fn coefficients(&self) -> js_sys::Float64Array {
        js_sys::Float64Array::from(self.session.coefficients().data.as_slice())
    }
}

pub mod single;
#[wasm_bindgen]
impl AnnyModel {
    /// Start a reusable pose session for repeated re-posing with fixed non-pose parameters.
    pub fn pose_session(&self, parameters_json: Option<String>) -> Result<AnnySession, JsValue> {
        let parameters: Parameters = parameters_json
            .as_deref()
            .map(serde_json::from_str)
            .transpose()
            .map_err(js_error)?
            .unwrap_or_default();
        let owner = Arc::clone(&self.model);
        // `owner` keeps the model alive for as long as the session borrows it, and a model is
        // immutable after construction, so widening the borrow to 'static here is sound.
        let model_ref: &'static Anny = unsafe { &*Arc::as_ptr(&owner) };
        Ok(AnnySession {
            _model: owner,
            session: model_ref.pose_session(&parameters).map_err(js_error)?,
        })
    }
    pub fn to_f32(&self) -> Result<single::AnnyModelF32, JsValue> {
        Ok(single::AnnyModelF32 {
            model: Arc::new(self.model.to_f32().map_err(js_error)?),
        })
    }
}

/// In-memory glTF authoring. Owned JS outputs survive memory growth and disposal.
#[wasm_bindgen]
pub struct GltfDocument {
    asset: anny_core::gltf_asset::GltfAsset,
}
#[wasm_bindgen]
impl GltfDocument {
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: &[u8]) -> Result<GltfDocument, JsValue> {
        Ok(Self {
            asset: anny_core::gltf_asset::GltfAsset::from_bytes(bytes).map_err(js_error)?,
        })
    }
    pub fn query(&self, request_json: &str) -> Result<String, JsValue> {
        let query = serde_json::from_str(request_json).map_err(js_error)?;
        serde_json::to_string(&self.asset.query(&query).map_err(js_error)?).map_err(js_error)
    }
    pub fn edit(&mut self, operations_json: &str) -> Result<String, JsValue> {
        let operations: Vec<anny_core::gltf_asset::GltfEdit> =
            serde_json::from_str(operations_json).map_err(js_error)?;
        serde_json::to_string(&self.asset.apply_edits(&operations).map_err(js_error)?)
            .map_err(js_error)
    }
    pub fn glb(&self) -> Result<js_sys::Uint8Array, JsValue> {
        Ok(js_sys::Uint8Array::from(
            self.asset.to_glb().map_err(js_error)?.as_slice(),
        ))
    }
}
