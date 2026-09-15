//! Stable C ABI. Pointer ownership and lifetimes are specified in include/anny.h.
//! No Rust containers, exceptions, or Rust ABI symbols cross this boundary.
use anny_core::{
    assets::AssetStore, Anny, AnnyConfig, ModelOutput, Parameters, PoseSession, Tensor,
};
use std::{
    cell::RefCell,
    ffi::{c_char, CStr, CString},
    panic::{catch_unwind, AssertUnwindSafe},
    ptr,
    sync::Arc,
};

pub struct AnnyModel {
    model: Arc<Anny>,
}
pub struct AnnyOutput {
    output: ModelOutput,
}
/// A reusable pose session over a model; see `anny_session_new`.
pub struct AnnySession {
    // SAFETY INVARIANT: Rust drops fields in declaration order. `session` contains the widened
    // reference, so it must be destroyed before `_model` releases the Arc allocation it borrows.
    session: PoseSession<'static>,
    /// Keeps the allocation the session evaluates against alive until after `session` is dropped.
    _model: Arc<Anny>,
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
            *out = Box::into_raw(Box::new(AnnyModel { model: Arc::new(m) }));
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
            *out = Box::into_raw(Box::new(AnnyModel { model: Arc::new(m) }));
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
            *out = Box::into_raw(Box::new(AnnyModel { model: Arc::new(m) }));
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
            *out = Box::into_raw(Box::new(AnnyModel {
                model: Arc::new(model),
            }));
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
                model: Arc::new(result.model),
            }));
        }
        Ok(())
    })
}

#[cfg(test)]
#[path = "../../anny-core/tests/common/mod.rs"]
mod fixture;

/// Create a reusable pose session for repeated re-posing with a fixed phenotype/local-change/facial
/// selection: those coefficients and the rest model are evaluated once here, and each later
/// `anny_session_update` pays only for the pose-dependent half (roughly half of
/// `anny_model_evaluate`). The session holds its own reference to the model, so the model handle may
/// be freed before the session.
/// # Safety
/// model is a live handle or null; parameters is null (defaults) or a valid NUL-terminated UTF-8
/// JSON string; out points to writable handle storage.
#[no_mangle]
pub unsafe extern "C" fn anny_session_new(
    model: *const AnnyModel,
    parameters: *const c_char,
    out: *mut *mut AnnySession,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return Err("null session handle".into());
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
        let owner = Arc::clone(&m.model);
        // The session evaluates through a reference into `owner`'s allocation: `owner` is kept in the
        // session so the allocation outlives it, and a model is immutable after construction, so
        // widening the borrow to 'static is sound.
        let model_ref: &'static Anny = unsafe { &*Arc::as_ptr(&owner) };
        let session = model_ref.pose_session(&p).map_err(|e| e.to_string())?;
        unsafe {
            *out = Box::into_raw(Box::new(AnnySession {
                _model: owner,
                session,
            }));
        }
        Ok(())
    })
}
/// Re-pose the session, reusing the coefficients and rest model it was created with.
/// # Safety
/// session is a live handle; pose_json is null (the zero pose) or a valid NUL-terminated UTF-8 JSON
/// pose document, the same object `anny_model_evaluate` takes in `pose_parameters`. Tensor views
/// taken from the session before this call are invalid afterwards.
#[no_mangle]
pub unsafe extern "C" fn anny_session_update(
    session: *mut AnnySession,
    pose_json: *const c_char,
) -> i32 {
    guard(|| {
        let s = unsafe { session.as_mut() }.ok_or("null session")?;
        let pose = if pose_json.is_null() {
            serde_json::Value::Null
        } else {
            serde_json::from_str(unsafe { text(pose_json)? }).map_err(|e| e.to_string())?
        };
        s.session.update(&pose).map_err(|e| e.to_string())?;
        Ok(())
    })
}
/// Read one array of the session's current result.
/// # Safety
/// session/name are valid live pointers; out points to writable view storage. Returned data is
/// read-only and borrowed until the session is freed or updated again.
#[no_mangle]
pub unsafe extern "C" fn anny_session_tensor(
    session: *const AnnySession,
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
        let s = unsafe { session.as_ref() }.ok_or("null session")?;
        let tensor = s
            .session
            .output()
            .get(unsafe { text(name)? })
            .map_err(|e| e.to_string())?;
        unsafe {
            *out = view(tensor);
        }
        Ok(())
    })
}
/// Read the coefficients the session was created with.
/// # Safety
/// session is a valid live pointer; out points to writable view storage. Returned data is read-only
/// and borrowed until the session is freed.
#[no_mangle]
pub unsafe extern "C" fn anny_session_coefficients(
    session: *const AnnySession,
    out: *mut AnnyTensorView,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return Err("null view".into());
        }
        unsafe {
            *out = AnnyTensorView::default();
        }
        let s = unsafe { session.as_ref() }.ok_or("null session")?;
        unsafe {
            *out = view(s.session.coefficients());
        }
        Ok(())
    })
}
/// # Safety
/// session is null or a live handle, freed exactly once, with no concurrent use.
#[no_mangle]
pub unsafe extern "C" fn anny_session_free(session: *mut AnnySession) {
    if !session.is_null() {
        unsafe {
            drop(Box::from_raw(session));
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
    #[test]
    fn secondary_ownership_and_prepared_roundtrip() {
        let model = Box::into_raw(Box::new(AnnyModel {
            model: Arc::new(fixture::tiny()),
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

/// Edit self-contained glTF/GLB bytes using gltf_asset::GltfEdit JSON. No model
/// handle is needed. Free the independent result with anny_bytes_free.
/// # Safety
/// bytes covers len readable bytes; operations_json is NUL-terminated UTF-8;
/// out points to writable, nonaliased handle storage.
#[no_mangle]
pub unsafe extern "C" fn anny_gltf_edit(
    bytes: *const u8,
    len: usize,
    operations_json: *const c_char,
    out: *mut *mut AnnyBytes,
) -> i32 {
    guard(|| {
        if out.is_null() {
            return Err("null bytes output".into());
        }
        unsafe {
            *out = ptr::null_mut();
        }
        if bytes.is_null() || len == 0 || len > 512 * 1024 * 1024 {
            return Err("invalid glTF byte buffer".into());
        }
        let input = unsafe { std::slice::from_raw_parts(bytes, len) };
        let result = anny_core::gltf_asset::edit_glb(input, unsafe { text(operations_json)? })
            .map_err(|e| e.to_string())?;
        unsafe {
            *out = Box::into_raw(Box::new(AnnyBytes { bytes: result }));
        }
        Ok(())
    })
}
/// Query self-contained glTF/GLB bytes; free returned UTF-8 with anny_string_free.
/// # Safety
/// bytes covers len readable bytes, request_json is NUL-terminated UTF-8 and
/// out points to writable, nonaliased pointer storage.
#[no_mangle]
pub unsafe extern "C" fn anny_gltf_query(
    bytes: *const u8,
    len: usize,
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
        if bytes.is_null() || len == 0 || len > 512 * 1024 * 1024 {
            return Err("invalid glTF byte buffer".into());
        }
        let input = unsafe { std::slice::from_raw_parts(bytes, len) };
        let result = anny_core::gltf_asset::query_glb(input, unsafe { text(request_json)? })
            .map_err(|e| e.to_string())?;
        let result = CString::new(result).map_err(|e| e.to_string())?;
        unsafe {
            *out = result.into_raw();
        }
        Ok(())
    })
}

#[cfg(test)]
mod gltf_tests {
    use super::*;
    #[test]
    fn standalone_gltf_bytes_and_text_are_owned_and_failures_clear_outputs() {
        let mut scene = anny_core::scene::Scene::new();
        scene
            .add_character(&fixture::tiny(), &Default::default(), &Default::default())
            .unwrap();
        let input = scene.to_glb().unwrap();
        let operations = CString::new("[]").unwrap();
        let request = CString::new(r#"{"operation":"describe"}"#).unwrap();
        let mut bytes = ptr::null_mut();
        assert_eq!(
            unsafe { anny_gltf_edit(input.as_ptr(), input.len(), operations.as_ptr(), &mut bytes) },
            0
        );
        drop(input);
        let mut text = ptr::null_mut();
        assert_eq!(
            unsafe {
                anny_gltf_query(
                    anny_bytes_data(bytes),
                    anny_bytes_len(bytes),
                    request.as_ptr(),
                    &mut text,
                )
            },
            0
        );
        unsafe {
            anny_bytes_free(bytes);
        }
        assert!(unsafe { CStr::from_ptr(text) }
            .to_str()
            .unwrap()
            .contains("meshes"));
        unsafe {
            anny_string_free(text);
        }
        bytes = ptr::dangling_mut();
        assert_ne!(
            unsafe { anny_gltf_edit(ptr::null(), 0, operations.as_ptr(), &mut bytes) },
            0
        );
        assert!(bytes.is_null());
    }
}

#[cfg(test)]
mod session_tests {
    use super::*;
    /// Compare every array of a session against the arrays of an output handle, name by name.
    fn assert_same_session(
        session: *const AnnySession,
        output: *const AnnyOutput,
        names: &[String],
    ) {
        for name in names {
            let cname = CString::new(name.as_str()).unwrap();
            let mut a = AnnyTensorView::default();
            let mut b = AnnyTensorView::default();
            assert_eq!(
                unsafe { anny_session_tensor(session, cname.as_ptr(), &mut a) },
                0,
                "session tensor {name}"
            );
            assert_eq!(
                unsafe { anny_output_tensor(output, cname.as_ptr(), &mut b) },
                0,
                "output tensor {name}"
            );
            assert_eq!(a.len, b.len, "{name} length");
            assert_eq!(a.rank, b.rank, "{name} rank");
            let left_shape = unsafe { std::slice::from_raw_parts(a.shape, a.rank) };
            let right_shape = unsafe { std::slice::from_raw_parts(b.shape, b.rank) };
            assert_eq!(left_shape, right_shape, "{name} shape");
            assert_eq!(a.kind, b.kind, "{name} kind");
            let left = unsafe { std::slice::from_raw_parts(a.data, a.len) };
            let right = unsafe { std::slice::from_raw_parts(b.data, b.len) };
            assert_eq!(left, right, "{name} data");
        }
    }
    #[test]
    fn session_matches_evaluate_and_outlives_the_model_handle() {
        // Names come from the core so every array the model produces is compared, not a sample.
        let reference = fixture::tiny().forward(&Parameters::default()).unwrap();
        let names: Vec<String> = reference.arrays.keys().cloned().collect();
        assert!(!names.is_empty());
        let model = Box::into_raw(Box::new(AnnyModel {
            model: Arc::new(fixture::tiny()),
        }));
        let mut output = ptr::null_mut();
        assert_eq!(
            unsafe { anny_model_evaluate(model, ptr::null(), &mut output) },
            0
        );
        let mut session = ptr::null_mut();
        assert_eq!(
            unsafe { anny_session_new(model, ptr::null(), &mut session) },
            0
        );
        // Before any update the session exposes the rest model, like a null-pose forward.
        assert_eq!(unsafe { anny_session_update(session, ptr::null()) }, 0);
        assert_same_session(session, output, &names);
        // The session holds its own reference, so freeing the caller's handle must not matter.
        unsafe {
            anny_model_free(model);
        }
        assert_eq!(unsafe { anny_session_update(session, ptr::null()) }, 0);
        assert_same_session(session, output, &names);
        let mut coefficients = AnnyTensorView::default();
        assert_eq!(
            unsafe { anny_session_coefficients(session, &mut coefficients) },
            0
        );
        assert!(coefficients.len > 0);
        unsafe {
            anny_session_free(session);
            anny_output_free(output);
        }
    }
    #[test]
    fn session_rejects_bad_input_and_stays_usable() {
        let model = Box::into_raw(Box::new(AnnyModel {
            model: Arc::new(fixture::tiny()),
        }));
        let mut session = ptr::null_mut();
        assert_eq!(
            unsafe { anny_session_new(model, ptr::null(), &mut session) },
            0
        );
        let mut view = AnnyTensorView::default();
        let name = CString::new("vertices").unwrap();
        // A pose the model cannot satisfy is refused, and leaves the session alone.
        let bad = CString::new(r#"{"root":[[0,0,0]]}"#).unwrap();
        assert_eq!(unsafe { anny_session_update(session, bad.as_ptr()) }, 1);
        assert!(!anny_last_error().is_null());
        assert_eq!(unsafe { anny_session_update(session, ptr::null()) }, 0);
        assert_eq!(
            unsafe { anny_session_tensor(session, name.as_ptr(), &mut view) },
            0
        );
        // Null handles and malformed documents are errors, never aborts.
        assert_eq!(
            unsafe { anny_session_update(ptr::null_mut(), ptr::null()) },
            1
        );
        assert_eq!(
            unsafe { anny_session_tensor(ptr::null(), name.as_ptr(), &mut view) },
            1
        );
        let mut scratch = ptr::null_mut();
        assert_eq!(
            unsafe { anny_session_new(model, ptr::null(), &mut scratch) },
            0
        );
        assert_eq!(
            unsafe { anny_session_new(ptr::null(), ptr::null(), &mut scratch) },
            1
        );
        assert!(scratch.is_null());
        assert_eq!(
            unsafe { anny_session_new(model, ptr::null(), ptr::null_mut()) },
            1
        );
        unsafe {
            anny_session_free(session);
            anny_session_free(ptr::null_mut());
            anny_model_free(model);
        }
    }
}
