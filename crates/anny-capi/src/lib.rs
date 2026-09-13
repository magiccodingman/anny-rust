//! Stable C ABI. Pointer ownership and lifetimes are specified in include/anny.h.
//! No Rust containers, exceptions, or Rust ABI symbols cross this boundary.
use anny_core::{assets::AssetStore, Anny, AnnyConfig, ModelOutput, Parameters, Tensor};
use std::{
    cell::RefCell,
    ffi::{c_char, CStr, CString},
    panic::{catch_unwind, AssertUnwindSafe},
    ptr,
};

pub struct AnnyModel {
    model: Anny,
}
pub struct AnnyOutput {
    output: ModelOutput,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct AnnyTensorView {
    pub data: *const f64,
    pub len: usize,
    pub shape: *const usize,
    pub rank: usize,
    pub kind: u32,
}
impl Default for AnnyTensorView {
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
thread_local! {static ERROR:RefCell<CString>=RefCell::new(CString::new("").unwrap());}
fn set_error(message: String) {
    ERROR.with(|s| *s.borrow_mut() = CString::new(message.replace('\0', " ")).unwrap());
}
fn guard(f: impl FnOnce() -> Result<(), String>) -> i32 {
    set_error(String::new());
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            set_error(e);
            1
        }
        Err(_) => {
            set_error("Rust panic caught at Anny ABI boundary".into());
            2
        }
    }
}
unsafe fn text<'a>(p: *const c_char) -> Result<&'a str, String> {
    if p.is_null() {
        return Err("null string argument".into());
    }
    unsafe { CStr::from_ptr(p) }
        .to_str()
        .map_err(|e| e.to_string())
}
unsafe fn config(p: *const c_char) -> Result<Option<AnnyConfig>, String> {
    if p.is_null() {
        Ok(None)
    } else {
        serde_json::from_str(unsafe { text(p)? })
            .map(Some)
            .map_err(|e| e.to_string())
    }
}
fn view(t: &Tensor) -> AnnyTensorView {
    AnnyTensorView {
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

#[no_mangle]
pub extern "C" fn anny_abi_version() -> u32 {
    1
}
/// Error text is borrowed thread-local memory until the next non-free ABI call.
#[no_mangle]
pub extern "C" fn anny_last_error() -> *const c_char {
    ERROR.with(|s| s.borrow().as_ptr())
}
/// # Safety
/// bytes must cover len readable bytes, config_json is null or a NUL-terminated
/// UTF-8 string, and out points to writable handle storage. No aliasing writes.
#[no_mangle]
pub unsafe extern "C" fn anny_model_from_bytes(
    bytes: *const u8,
    len: usize,
    config_json: *const c_char,
    out: *mut *mut AnnyModel,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return Err("null output handle".into());
        }
        unsafe {
            *out = ptr::null_mut();
        }
        if bytes.is_null() || len == 0 || len > isize::MAX as usize {
            return Err("invalid model byte buffer".into());
        }
        let data = unsafe { std::slice::from_raw_parts(bytes, len) };
        let m =
            Anny::from_bytes(data, unsafe { config(config_json)? }).map_err(|e| e.to_string())?;
        unsafe {
            *out = Box::into_raw(Box::new(AnnyModel { model: m }));
        }
        Ok(())
    })
}
/// # Safety
/// path/config_json are valid NUL-terminated UTF-8 strings (config may be null).
/// out points to writable handle storage and does not alias an input.
#[no_mangle]
pub unsafe extern "C" fn anny_model_load(
    path: *const c_char,
    config_json: *const c_char,
    out: *mut *mut AnnyModel,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return Err("null output handle".into());
        }
        unsafe {
            *out = ptr::null_mut();
        }
        let m = Anny::load(unsafe { text(path)? }, unsafe { config(config_json)? })
            .map_err(|e| e.to_string())?;
        unsafe {
            *out = Box::into_raw(Box::new(AnnyModel { model: m }));
        }
        Ok(())
    })
}
/// # Safety
/// assets/config_json are valid NUL-terminated UTF-8 strings (config may be null).
/// out points to writable handle storage and does not alias an input.
#[no_mangle]
pub unsafe extern "C" fn anny_model_build(
    assets: *const c_char,
    config_json: *const c_char,
    out: *mut *mut AnnyModel,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return Err("null output handle".into());
        }
        unsafe {
            *out = ptr::null_mut();
        }
        let c = unsafe { config(config_json)? }.unwrap_or_default();
        let m = AssetStore::new(unsafe { text(assets)? })
            .build(&c)
            .map_err(|e| e.to_string())?;
        unsafe {
            *out = Box::into_raw(Box::new(AnnyModel { model: m }));
        }
        Ok(())
    })
}
/// # Safety
/// model must be null or a live handle returned by this library, freed exactly
/// once, with no concurrent use and no borrowed views used after this call.
#[no_mangle]
pub unsafe extern "C" fn anny_model_free(model: *mut AnnyModel) {
    if !model.is_null() {
        unsafe {
            drop(Box::from_raw(model));
        }
    }
}
/// # Safety
/// model is live; parameters is null or a valid NUL-terminated UTF-8 JSON string;
/// out points to writable handle storage. Evaluation does not mutate the model.
#[no_mangle]
pub unsafe extern "C" fn anny_model_evaluate(
    model: *const AnnyModel,
    parameters: *const c_char,
    out: *mut *mut AnnyOutput,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return Err("null output handle".into());
        }
        unsafe {
            *out = ptr::null_mut();
        }
        let m = unsafe { model.as_ref() }.ok_or("null model")?;
        let p = if parameters.is_null() {
            Parameters::default()
        } else {
            serde_json::from_str(unsafe { text(parameters)? }).map_err(|e| e.to_string())?
        };
        let output = m.model.forward(&p).map_err(|e| e.to_string())?;
        unsafe {
            *out = Box::into_raw(Box::new(AnnyOutput { output }));
        }
        Ok(())
    })
}
/// # Safety
/// output must be null or a live handle, freed exactly once with no concurrent
/// use. All borrowed output tensor views become invalid after this call.
#[no_mangle]
pub unsafe extern "C" fn anny_output_free(output: *mut AnnyOutput) {
    if !output.is_null() {
        unsafe {
            drop(Box::from_raw(output));
        }
    }
}
/// # Safety
/// model/name are valid live pointers. out points to writable view storage.
/// Returned data is read-only and borrowed until model is freed.
#[no_mangle]
pub unsafe extern "C" fn anny_model_tensor(
    model: *const AnnyModel,
    name: *const c_char,
    out: *mut AnnyTensorView,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return Err("null view".into());
        }
        unsafe {
            *out = AnnyTensorView::default();
        }
        let m = unsafe { model.as_ref() }.ok_or("null model")?;
        let tensor = m
            .model
            .data
            .get(unsafe { text(name)? })
            .map_err(|e| e.to_string())?;
        unsafe {
            *out = view(tensor);
        }
        Ok(())
    })
}
/// # Safety
/// output/name are valid live pointers. out points to writable view storage.
/// Returned data is read-only and borrowed until output is freed.
#[no_mangle]
pub unsafe extern "C" fn anny_output_tensor(
    output: *const AnnyOutput,
    name: *const c_char,
    out: *mut AnnyTensorView,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return Err("null view".into());
        }
        unsafe {
            *out = AnnyTensorView::default();
        }
        let m = unsafe { output.as_ref() }.ok_or("null output")?;
        let tensor = m
            .output
            .get(unsafe { text(name)? })
            .map_err(|e| e.to_string())?;
        unsafe {
            *out = view(tensor);
        }
        Ok(())
    })
}
/// # Safety
/// model is live and out points to writable string pointer storage.
/// The resulting UTF-8 NUL-terminated string must be freed with anny_string_free.
#[no_mangle]
pub unsafe extern "C" fn anny_model_describe(
    model: *const AnnyModel,
    out: *mut *mut c_char,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return Err("null string output".into());
        }
        unsafe {
            *out = ptr::null_mut();
        }
        let m = unsafe { model.as_ref() }.ok_or("null model")?;
        let s = CString::new(m.model.describe().to_string()).map_err(|e| e.to_string())?;
        unsafe {
            *out = s.into_raw();
        }
        Ok(())
    })
}
/// # Safety
/// s is null or a string allocated by anny_model_describe and not yet freed.
#[no_mangle]
pub unsafe extern "C" fn anny_string_free(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            drop(CString::from_raw(s));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn null_arguments_are_errors() {
        let mut handle = ptr::null_mut();
        let status = unsafe { anny_model_from_bytes(ptr::null(), 0, ptr::null(), &mut handle) };
        assert_eq!(status, 1);
        assert!(handle.is_null());
        assert!(!anny_last_error().is_null());
    }
    #[test]
    fn errors_are_thread_local() {
        set_error("parent".into());
        std::thread::spawn(|| {
            set_error("child".into());
            assert_eq!(
                unsafe { CStr::from_ptr(anny_last_error()) }
                    .to_str()
                    .unwrap(),
                "child"
            );
        })
        .join()
        .unwrap();
        assert_eq!(
            unsafe { CStr::from_ptr(anny_last_error()) }
                .to_str()
                .unwrap(),
            "parent"
        );
    }
}
