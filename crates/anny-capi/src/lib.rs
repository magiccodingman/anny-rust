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
/// s is null or a string allocated by this library and not yet freed.
#[no_mangle]
pub unsafe extern "C" fn anny_string_free(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            drop(CString::from_raw(s));
        }
    }
}

/// Owned serialized data. Accessors borrow its memory until anny_bytes_free.
pub struct AnnyBytes {
    bytes: Vec<u8>,
}
/// Export a character to GLB. options_json follows scene::CharacterExport.
/// # Safety
/// model is live, strings are null/default or NUL-terminated UTF-8, and out
/// points to writable handle storage. The output is owned independently.
#[no_mangle]
pub unsafe extern "C" fn anny_model_export_glb(
    model: *const AnnyModel,
    parameters_json: *const c_char,
    options_json: *const c_char,
    out: *mut *mut AnnyBytes,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return Err("null byte output handle".into());
        }
        unsafe {
            *out = ptr::null_mut();
        }
        let m = unsafe { model.as_ref() }.ok_or("null model")?;
        let p: Parameters = if parameters_json.is_null() {
            Parameters::default()
        } else {
            serde_json::from_str(unsafe { text(parameters_json)? }).map_err(|e| e.to_string())?
        };
        let o: anny_core::scene::CharacterExport = if options_json.is_null() {
            Default::default()
        } else {
            serde_json::from_str(unsafe { text(options_json)? }).map_err(|e| e.to_string())?
        };
        let mut scene = anny_core::scene::Scene::new();
        scene
            .add_character(&m.model, &p, &o)
            .map_err(|e| e.to_string())?;
        let bytes = scene.to_glb().map_err(|e| e.to_string())?;
        unsafe {
            *out = Box::into_raw(Box::new(AnnyBytes { bytes }));
        }
        Ok(())
    })
}
/// # Safety
/// bytes is null or a live AnnyBytes. Returned data is borrowed read-only.
#[no_mangle]
pub unsafe extern "C" fn anny_bytes_data(bytes: *const AnnyBytes) -> *const u8 {
    unsafe { bytes.as_ref() }.map_or(ptr::null(), |b| b.bytes.as_ptr())
}
/// # Safety
/// bytes is null or a live AnnyBytes.
#[no_mangle]
pub unsafe extern "C" fn anny_bytes_len(bytes: *const AnnyBytes) -> usize {
    unsafe { bytes.as_ref() }.map_or(0, |b| b.bytes.len())
}
/// # Safety
/// bytes is null or a live uniquely owned handle, freed exactly once with no
/// concurrent readers or views used afterward.
#[no_mangle]
pub unsafe extern "C" fn anny_bytes_free(bytes: *mut AnnyBytes) {
    if !bytes.is_null() {
        unsafe {
            drop(Box::from_raw(bytes));
        }
    }
}

/// Execute a portable secondary request (measure/keypoints/pose/fit/sample/collision).
/// # Safety
/// model is live, request_json is NUL-terminated UTF-8, and out is writable.
/// Returned text is owned by the caller and freed with anny_string_free.
#[no_mangle]
pub unsafe extern "C" fn anny_model_query(
    model: *const AnnyModel,
    request_json: *const c_char,
    out: *mut *mut c_char,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return Err("null text output".into());
        }
        unsafe {
            *out = ptr::null_mut();
        }
        let m = unsafe { model.as_ref() }.ok_or("null model")?;
        let result = anny_core::operations::execute_json(&m.model, unsafe { text(request_json)? })
            .map_err(|e| e.to_string())?;
        let result = CString::new(result).map_err(|e| e.to_string())?;
        unsafe {
            *out = result.into_raw();
        }
        Ok(())
    })
}
/// Apply an explicit ModelData authoring pipeline to an independent model.
/// # Safety
/// model is live, operations_json is NUL-terminated UTF-8, out is writable and
/// does not alias model storage. Existing model/views remain unchanged.
#[no_mangle]
pub unsafe extern "C" fn anny_model_transform(
    model: *const AnnyModel,
    operations_json: *const c_char,
    out: *mut *mut AnnyModel,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return Err("null model output".into());
        }
        unsafe {
            *out = ptr::null_mut();
        }
        let m = unsafe { model.as_ref() }.ok_or("null model")?;
        let operations = serde_json::from_str::<Vec<anny_core::transforms::Transform>>(unsafe {
            text(operations_json)?
        })
        .map_err(|e| e.to_string())?;
        let model = anny_core::transforms::apply_pipeline(&m.model, &operations)
            .map_err(|e| e.to_string())?;
        unsafe {
            *out = Box::into_raw(Box::new(AnnyModel { model }));
        }
        Ok(())
    })
}
/// Serialize a portable prepared model with its configuration.
/// # Safety
/// model is live and out points to writable handle storage.
#[no_mangle]
pub unsafe extern "C" fn anny_model_prepared_bytes(
    model: *const AnnyModel,
    out: *mut *mut AnnyBytes,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return Err("null bytes output".into());
        }
        unsafe {
            *out = ptr::null_mut();
        }
        let m = unsafe { model.as_ref() }.ok_or("null model")?;
        let bytes = m
            .model
            .data
            .archive(Some(&m.model.config))
            .and_then(|a| a.to_bytes())
            .map_err(|e| e.to_string())?;
        unsafe {
            *out = Box::into_raw(Box::new(AnnyBytes { bytes }));
        }
        Ok(())
    })
}
/// Transfer pose between compatible rest meshes/rigs, returning parameter JSON.
/// # Safety
/// Both models are live (may be identical). Strings are NUL-terminated UTF-8,
/// parameters_json may be null/default, and out points to writable text storage.
#[no_mangle]
pub unsafe extern "C" fn anny_model_transfer_pose(
    source: *const AnnyModel,
    target: *const AnnyModel,
    parameters_json: *const c_char,
    mode: *const c_char,
    out: *mut *mut c_char,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return Err("null text output".into());
        }
        unsafe {
            *out = ptr::null_mut();
        }
        let a = unsafe { source.as_ref() }.ok_or("null source model")?;
        let b = unsafe { target.as_ref() }.ok_or("null target model")?;
        let params: Parameters = if parameters_json.is_null() {
            Parameters::default()
        } else {
            serde_json::from_str(unsafe { text(parameters_json)? }).map_err(|e| e.to_string())?
        };
        let mode: anny_core::PoseParameterization =
            serde_json::from_value(serde_json::json!(unsafe { text(mode)? }))
                .map_err(|e| e.to_string())?;
        let tensor = anny_core::tools::transfer_pose_parameters(&a.model, &b.model, &params, mode)
            .map_err(|e| e.to_string())?;
        let result=CString::new(serde_json::json!({"pose_parameterization":mode,"pose_parameters":tensor.nested_json()}).to_string()).map_err(|e|e.to_string())?;
        unsafe {
            *out = result.into_raw();
        }
        Ok(())
    })
}

/// Construct with an optional filesystem cache (null disables, "auto" uses OS defaults).
/// # Safety
/// assets is NUL-terminated UTF-8, config/cache are null or valid UTF-8 strings,
/// and out points to writable model-handle storage.
#[no_mangle]
pub unsafe extern "C" fn anny_model_build_cached(
    assets: *const c_char,
    config_json: *const c_char,
    cache_directory: *const c_char,
    out: *mut *mut AnnyModel,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return Err("null model output".into());
        }
        unsafe {
            *out = ptr::null_mut();
        }
        let config = unsafe { config(config_json)? }.unwrap_or_default();
        let store = AssetStore::new(unsafe { text(assets)? });
        let cache = if cache_directory.is_null() {
            anny_core::cache::ModelCache::disabled()
        } else {
            match unsafe { text(cache_directory)? } {
                "auto" => anny_core::cache::ModelCache::automatic().map_err(|e| e.to_string())?,
                path => anny_core::cache::ModelCache::directory(path),
            }
        };
        let result = cache
            .load_or_build(&store, &config)
            .map_err(|e| e.to_string())?;
        unsafe {
            *out = Box::into_raw(Box::new(AnnyModel {
                model: result.model,
            }));
        }
        Ok(())
    })
}

#[cfg(test)]
#[path = "../../anny-core/tests/common/mod.rs"]
mod fixture;

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
    #[test]
    fn secondary_ownership_and_prepared_roundtrip() {
        let model = Box::into_raw(Box::new(AnnyModel {
            model: fixture::tiny(),
        }));
        let request = CString::new(r#"{"operation":"pose-convert","mode":"world"}"#).unwrap();
        let mut result = ptr::null_mut();
        assert_eq!(
            unsafe { anny_model_query(model, request.as_ptr(), &mut result) },
            0
        );
        let text = unsafe { CStr::from_ptr(result) }.to_str().unwrap();
        assert!(text.contains("pose_parameters"));
        unsafe {
            anny_string_free(result);
        }
        let operations = CString::new(r#"[{"op":"triangulate"}]"#).unwrap();
        let mut modified = ptr::null_mut();
        assert_eq!(
            unsafe { anny_model_transform(model, operations.as_ptr(), &mut modified) },
            0
        );
        let mut bytes = ptr::null_mut();
        assert_eq!(
            unsafe { anny_model_prepared_bytes(modified, &mut bytes) },
            0
        );
        let mut copy = ptr::null_mut();
        assert_eq!(
            unsafe {
                anny_model_from_bytes(
                    anny_bytes_data(bytes),
                    anny_bytes_len(bytes),
                    ptr::null(),
                    &mut copy,
                )
            },
            0
        );
        unsafe {
            anny_bytes_free(bytes);
            anny_model_free(modified);
            anny_model_free(copy);
            anny_model_free(model);
        }
    }
}

pub mod single;
