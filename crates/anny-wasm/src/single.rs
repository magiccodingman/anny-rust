//! Owned JavaScript Float32Arrays; no views invalidated by WASM memory growth.
use super::js_error;
use anny_core::{AnnyF32, ModelOutputF32, Parameters};
use wasm_bindgen::prelude::*;
#[wasm_bindgen]
pub struct AnnyModelF32 {
    pub(crate) model: AnnyF32,
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
            model: AnnyF32::from_bytes(bytes, config).map_err(js_error)?,
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
