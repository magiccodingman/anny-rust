//! Browser wrapper. Supply prepared ModelData bytes; filesystem/asset construction
//! and networking are deliberately left to the host, outside the browser runtime.
use anny_core::{Anny, AnnyConfig, ModelOutput, Parameters, Tensor};
use wasm_bindgen::prelude::*;
fn js_error(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}
#[wasm_bindgen]
pub struct AnnyModel {
    model: Anny,
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
            model: Anny::from_bytes(bytes, config).map_err(js_error)?,
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
            model: anny_core::transforms::apply_pipeline(&self.model, &operations)
                .map_err(js_error)?,
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
