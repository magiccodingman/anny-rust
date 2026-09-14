//! Additive ABI-1 single-precision handles. Never reinterpret a v1 double view.
use super::*;
use anny_core::{AnnyF32, ModelOutputF32, TensorF32};

pub struct AnnyModelF32 {
    model: AnnyF32,
}
pub struct AnnyOutputF32 {
    output: ModelOutputF32,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct AnnyTensorViewF32 {
    pub data: *const f32,
    pub len: usize,
    pub shape: *const usize,
    pub rank: usize,
    pub kind: u32,
}
impl Default for AnnyTensorViewF32 {
    fn default() -> Self {
        Self {
            data: ptr::null(),
            len: 0,
            shape: ptr::null(),
            rank: 0,
            kind: 0,
        }
    }
}
fn view32(t: &TensorF32) -> AnnyTensorViewF32 {
    AnnyTensorViewF32 {
        data: t.data.as_ptr(),
        len: t.data.len(),
        shape: t.shape.as_ptr(),
        rank: t.shape.len(),
        kind: match t.kind {
            anny_core::tensor::Kind::Float => 0,
            anny_core::tensor::Kind::Index => 1,
            anny_core::tensor::Kind::Bool => 2,
        },
    }
}
/// Create an independent single-precision copy. The source handle is not retained.
/// # Safety
/// source is a live f64 model; out is writable and aliases no model data.
#[no_mangle]
pub unsafe extern "C" fn anny_model_to_f32(
    source: *const AnnyModel,
    out: *mut *mut AnnyModelF32,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return Err("null f32 model output".into());
        }
        unsafe {
            *out = ptr::null_mut();
        }
        let source = unsafe { source.as_ref() }.ok_or("null model")?;
        let model = source.model.to_f32().map_err(|e| e.to_string())?;
        unsafe {
            *out = Box::into_raw(Box::new(AnnyModelF32 { model }));
        }
        Ok(())
    })
}
/// # Safety
/// bytes covers len readable bytes; config is null or NUL-terminated UTF-8;
/// out is writable handle storage and aliases no input.
#[no_mangle]
pub unsafe extern "C" fn anny_model_f32_from_bytes(
    bytes: *const u8,
    len: usize,
    config_json: *const c_char,
    out: *mut *mut AnnyModelF32,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return Err("null f32 model output".into());
        }
        unsafe {
            *out = ptr::null_mut();
        }
        if bytes.is_null() || len == 0 || len > isize::MAX as usize {
            return Err("invalid model byte buffer".into());
        }
        let bytes = unsafe { std::slice::from_raw_parts(bytes, len) };
        let model = AnnyF32::from_bytes(bytes, unsafe { config(config_json)? })
            .map_err(|e| e.to_string())?;
        unsafe {
            *out = Box::into_raw(Box::new(AnnyModelF32 { model }));
        }
        Ok(())
    })
}
/// # Safety
/// model is live; parameters is null or NUL-terminated UTF-8; out is writable.
#[no_mangle]
pub unsafe extern "C" fn anny_model_f32_evaluate(
    model: *const AnnyModelF32,
    parameters: *const c_char,
    out: *mut *mut AnnyOutputF32,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return Err("null f32 result output".into());
        }
        unsafe {
            *out = ptr::null_mut();
        }
        let model = unsafe { model.as_ref() }.ok_or("null f32 model")?;
        let p = if parameters.is_null() {
            Parameters::default()
        } else {
            serde_json::from_str(unsafe { text(parameters)? }).map_err(|e| e.to_string())?
        };
        let output = model.model.forward(&p).map_err(|e| e.to_string())?;
        unsafe {
            *out = Box::into_raw(Box::new(AnnyOutputF32 { output }));
        }
        Ok(())
    })
}
/// # Safety
/// model is live, name is NUL-terminated UTF-8, out is writable. View is borrowed
/// until model is freed. Never use a double-view layout to read this data.
#[no_mangle]
pub unsafe extern "C" fn anny_model_f32_tensor(
    model: *const AnnyModelF32,
    name: *const c_char,
    out: *mut AnnyTensorViewF32,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return Err("null f32 tensor view".into());
        }
        unsafe {
            *out = AnnyTensorViewF32::default();
        }
        let m = unsafe { model.as_ref() }.ok_or("null f32 model")?;
        let t = m
            .model
            .data()
            .get(unsafe { text(name)? })
            .map_err(|e| e.to_string())?;
        unsafe {
            *out = view32(t);
        }
        Ok(())
    })
}
/// # Safety
/// output is live, name is NUL-terminated UTF-8, out is writable. View is borrowed
/// until output is freed, not tied to the model lifetime.
#[no_mangle]
pub unsafe extern "C" fn anny_output_f32_tensor(
    output: *const AnnyOutputF32,
    name: *const c_char,
    out: *mut AnnyTensorViewF32,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return Err("null f32 tensor view".into());
        }
        unsafe {
            *out = AnnyTensorViewF32::default();
        }
        let o = unsafe { output.as_ref() }.ok_or("null f32 output")?;
        let t = o
            .output
            .get(unsafe { text(name)? })
            .map_err(|e| e.to_string())?;
        unsafe {
            *out = view32(t);
        }
        Ok(())
    })
}
/// # Safety
/// model is live; out is writable. Free returned bytes with anny_bytes_free.
#[no_mangle]
pub unsafe extern "C" fn anny_model_f32_prepared_bytes(
    model: *const AnnyModelF32,
    out: *mut *mut AnnyBytes,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return Err("null byte output".into());
        }
        unsafe {
            *out = ptr::null_mut();
        }
        let m = unsafe { model.as_ref() }.ok_or("null f32 model")?;
        let data = m.model.to_bytes().map_err(|e| e.to_string())?;
        unsafe {
            *out = Box::into_raw(Box::new(AnnyBytes { bytes: data }));
        }
        Ok(())
    })
}
/// # Safety
/// model is null or a live f32 model, freed once, with no concurrent use.
#[no_mangle]
pub unsafe extern "C" fn anny_model_f32_free(model: *mut AnnyModelF32) {
    if !model.is_null() {
        unsafe {
            drop(Box::from_raw(model));
        }
    }
}
/// # Safety
/// output is null or a live f32 result, freed once, with no concurrent view use.
#[no_mangle]
pub unsafe extern "C" fn anny_output_f32_free(output: *mut AnnyOutputF32) {
    if !output.is_null() {
        unsafe {
            drop(Box::from_raw(output));
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn owned_float_results_and_errors() {
        let m = Box::into_raw(Box::new(AnnyModel {
            model: fixture::tiny(),
        }));
        let mut f = ptr::null_mut();
        let mut o = ptr::null_mut();
        assert_eq!(unsafe { anny_model_to_f32(m, &mut f) }, 0);
        unsafe {
            anny_model_free(m);
        }
        assert_eq!(
            unsafe { anny_model_f32_evaluate(f, ptr::null(), &mut o) },
            0
        );
        unsafe {
            anny_model_f32_free(f);
        }
        let mut v = AnnyTensorViewF32::default();
        let name = CString::new("vertices").unwrap();
        assert_eq!(
            unsafe { anny_output_f32_tensor(o, name.as_ptr(), &mut v) },
            0
        );
        assert_eq!(v.len, 9);
        assert!(unsafe { std::slice::from_raw_parts(v.data, v.len) }
            .iter()
            .all(|x| x.is_finite()));
        unsafe {
            anny_output_f32_free(o);
        }
        assert_eq!(unsafe { anny_model_to_f32(ptr::null(), &mut f) }, 1);
        assert!(f.is_null());
    }
}
