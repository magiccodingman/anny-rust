//! Owned JavaScript Float32Arrays; no views invalidated by WASM memory growth.
use super::js_error;
use anny_core::{AnnyF32, ModelOutputF32, Parameters, PoseSessionF32};
use std::sync::Arc;
use wasm_bindgen::prelude::*;
#[wasm_bindgen]
pub struct AnnyModelF32 {
    pub(crate) model: Arc<AnnyF32>,
}
#[wasm_bindgen]
pub struct AnnyResultF32 {
    output: ModelOutputF32,
}
#[wasm_bindgen]
impl AnnyModelF32 {
    #[wasm_bindgen(constructor)]
    pub fn new(bytes: &[u8], config_json: Option<String>) -> Result<AnnyModelF32, JsValue> {
        let config = config_json
            .as_deref()
            .map(serde_json::from_str)
            .transpose()
            .map_err(js_error)?;
        Ok(Self {
            model: Arc::new(AnnyF32::from_bytes(bytes, config).map_err(js_error)?),
        })
    }
    pub fn evaluate(&self, parameters_json: Option<String>) -> Result<AnnyResultF32, JsValue> {
        let p: Parameters = parameters_json
            .as_deref()
            .map(serde_json::from_str)
            .transpose()
            .map_err(js_error)?
            .unwrap_or_default();
        Ok(AnnyResultF32 {
            output: self.model.forward(&p).map_err(js_error)?,
        })
    }
    pub fn describe(&self) -> String {
        self.model.describe().to_string()
    }
    pub fn tensor(&self, name: &str) -> Result<js_sys::Float32Array, JsValue> {
        Ok(js_sys::Float32Array::from(
            self.model
                .data()
                .get(name)
                .map_err(js_error)?
                .data
                .as_slice(),
        ))
    }
    pub fn indices(&self, name: &str) -> Result<js_sys::Uint32Array, JsValue> {
        let t = self.model.data().get(name).map_err(js_error)?;
        let values = t
            .data
            .iter()
            .map(|&x| {
                if !x.is_finite() || x < 0. || x as f64 > u32::MAX as f64 || x.fract() != 0. {
                    Err(js_error("tensor cannot be represented as u32 indices"))
                } else {
                    Ok(x as u32)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(js_sys::Uint32Array::from(values.as_slice()))
    }
    /// Start a reusable f32 pose session; see the f64 `AnnySession` for the ownership rules.
    pub fn pose_session(&self, parameters_json: Option<String>) -> Result<AnnySessionF32, JsValue> {
        let parameters: Parameters = parameters_json
            .as_deref()
            .map(serde_json::from_str)
            .transpose()
            .map_err(js_error)?
            .unwrap_or_default();
        let owner = Arc::clone(&self.model);
        // `owner` keeps the model alive for as long as the session borrows it; a model is immutable
        // after construction, so widening the borrow to 'static here is sound.
        let model_ref: &'static AnnyF32 = unsafe { &*Arc::as_ptr(&owner) };
        Ok(AnnySessionF32 {
            _model: owner,
            session: model_ref.pose_session(&parameters).map_err(js_error)?,
        })
    }
    pub fn prepared_bytes(&self) -> Result<js_sys::Uint8Array, JsValue> {
        Ok(js_sys::Uint8Array::from(
            self.model.to_bytes().map_err(js_error)?.as_slice(),
        ))
    }
}
#[wasm_bindgen]
impl AnnyResultF32 {
    pub fn tensor(&self, name: &str) -> Result<js_sys::Float32Array, JsValue> {
        Ok(js_sys::Float32Array::from(
            self.output.get(name).map_err(js_error)?.data.as_slice(),
        ))
    }
    pub fn shape(&self, name: &str) -> Result<js_sys::Uint32Array, JsValue> {
        let shape = self
            .output
            .get(name)
            .map_err(js_error)?
            .shape
            .iter()
            .map(|&x| u32::try_from(x).map_err(js_error))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(js_sys::Uint32Array::from(shape.as_slice()))
    }
}

/// A reusable f32 pose session: `update` pays only for the pose, and the session keeps its own
/// reference to the model, so the model object may be dropped in JavaScript.
#[wasm_bindgen]
pub struct AnnySessionF32 {
    _model: Arc<AnnyF32>,
    session: PoseSessionF32<'static>,
}
#[wasm_bindgen]
impl AnnySessionF32 {
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
    pub fn tensor(&self, name: &str) -> Result<js_sys::Float32Array, JsValue> {
        Ok(js_sys::Float32Array::from(
            self.session
                .output()
                .get(name)
                .map_err(js_error)?
                .data
                .as_slice(),
        ))
    }
    pub fn shape(&self, name: &str) -> Result<js_sys::Uint32Array, JsValue> {
        let t = self.session.output().get(name).map_err(js_error)?;
        let values = t
            .shape
            .iter()
            .map(|&n| u32::try_from(n).map_err(js_error))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(js_sys::Uint32Array::from(values.as_slice()))
    }
    /// Owned copy of the coefficients this session was created with.
    pub fn coefficients(&self) -> js_sys::Float32Array {
        js_sys::Float32Array::from(self.session.coefficients().data.as_slice())
    }
}
