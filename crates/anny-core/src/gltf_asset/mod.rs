//! Owned base-glTF authoring document. Unlike geometry-only import, this retains
//! materials, images, skins, morph targets, animation channels and metadata.
//! Only glTF 2.0 without extensions and supported triangle geometry is accepted.
//! Image payloads remain opaque PNG/JPEG bytes; no renderer is embedded.
mod animation;
mod authoring;
pub use animation::{AnimationChannel, AnimationClip, AnimationPath, Interpolation};
pub use authoring::{
    AlphaMode, MorphDeltas, PbrMaterial, TextureFilter, TextureOptions, TextureReference, WrapMode,
};

use crate::{
    ensure,
    mesh_io::{parse_gltf, Gltf},
    scene::{Scene, SurfaceMesh},
    Error, Result,
};
use base64::Engine;
use serde_json::{json, Value};
use std::path::{Component, Path};

const LIMIT: usize = 512 * 1024 * 1024;
fn bad(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}
pub(crate) fn index(v: &Value, key: &str) -> Result<usize> {
    v.get(key)
        .and_then(Value::as_u64)
        .and_then(|i| usize::try_from(i).ok())
        .ok_or_else(|| bad(format!("missing/invalid glTF {key}")))
}
fn no_extensions(v: &Value) -> bool {
    match v {
        Value::Object(map) => map.iter().all(|(k, v)| {
            if k == "extensions" {
                v.as_object().is_some_and(|m| m.is_empty())
            } else if k == "extensionsUsed" || k == "extensionsRequired" {
                v.as_array().is_some_and(Vec::is_empty)
            } else if k == "extras" {
                true
            } else {
                no_extensions(v)
            }
        }),
        Value::Array(a) => a.iter().all(no_extensions),
        _ => true,
    }
}
fn read_limited(path: &Path) -> Result<Vec<u8>> {
    use std::io::Read;
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(LIMIT as u64 + 1)
        .read_to_end(&mut bytes)?;
    ensure(bytes.len() <= LIMIT, "glTF file exceeds byte limit")?;
    Ok(bytes)
}
fn uri_bytes(uri: &str, dir: Option<&Path>) -> Result<Vec<u8>> {
    if uri.starts_with("data:") {
        let (header, data) = uri
            .split_once(',')
            .ok_or_else(|| bad("invalid image data URI"))?;
        ensure(header.ends_with(";base64"), "image URI must be base64")?;
        ensure(data.len() <= LIMIT * 4 / 3 + 4, "image URI too large")?;
        return base64::engine::general_purpose::STANDARD
            .decode(data)
            .map_err(|_| bad("invalid base64 image"));
    }
    ensure(
        !uri.is_empty() && !uri.contains([':', '%', '\\', '?', '#']),
        "image URI must be a plain relative path",
    )?;
    let path = Path::new(uri);
    ensure(
        path.components()
            .all(|c| matches!(c, Component::Normal(_) | Component::CurDir)),
        "image URI escapes document",
    )?;
    let root = dir
        .ok_or_else(|| bad("external image needs a filesystem document path"))?
        .canonicalize()?;
    let path = root.join(path).canonicalize()?;
    ensure(path.starts_with(root), "image symlink escapes document")?;
    read_limited(&path)
}
pub(crate) fn mime(bytes: &[u8]) -> Result<&'static str> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Ok("image/png")
    } else if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        Ok("image/jpeg")
    } else {
        Err(bad("image must contain a PNG or JPEG payload"))
    }
}

#[derive(Clone, Debug)]
pub struct GltfAsset {
    pub(crate) graph: Gltf,
}
impl GltfAsset {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::parse(bytes, None)
    }
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        Self::parse(&read_limited(path)?, path.parent())
    }
    pub fn from_scene(scene: &Scene) -> Result<Self> {
        Self::from_bytes(&scene.to_glb()?)
    }
    fn parse(bytes: &[u8], dir: Option<&Path>) -> Result<Self> {
        let mut graph = parse_gltf(bytes, dir)?;
        ensure(
            no_extensions(&graph.root),
            "authoring currently supports base glTF without extensions",
        )?;
        let images = match graph.root.get("images") {
            Some(v) => v
                .as_array()
                .ok_or_else(|| bad("images must be an array"))?
                .clone(),
            None => Vec::new(),
        };
        let mut total: usize = graph.buffers.iter().map(Vec::len).sum();
        for (i, image) in images.iter().enumerate() {
            ensure(
                image.get("uri").is_some() != image.get("bufferView").is_some(),
                "image requires exactly one URI or bufferView",
            )?;
            let data = if let Some(uri) = image.get("uri") {
                uri_bytes(uri.as_str().ok_or_else(|| bad("invalid image URI"))?, dir)?
            } else {
                graph.view(index(image, "bufferView")?)?.0.to_vec()
            };
            let actual_mime = mime(&data)?;
            if let Some(m) = image.get("mimeType") {
                ensure(
                    m.as_str() == Some(actual_mime),
                    "image MIME/payload mismatch",
                )?;
            }
            if image.get("uri").is_none() {
                continue;
            }
            total = total
                .checked_add(data.len())
                .ok_or_else(|| bad("image byte overflow"))?;
            ensure(total <= LIMIT, "combined buffers/images exceed byte limit")?;
            let buffer = graph.buffers.len();
            let length = data.len();
            graph.buffers.push(data);
            graph.root["buffers"]
                .as_array_mut()
                .unwrap()
                .push(json!({"byteLength":length}));
            let views = array_mut(&mut graph.root, "bufferViews")?;
            let view = views.len();
            views.push(json!({"buffer":buffer,"byteLength":length}));
            let item = graph.root["images"][i]
                .as_object_mut()
                .ok_or_else(|| bad("invalid image object"))?;
            item.remove("uri");
            item.insert("bufferView".into(), json!(view));
            item.insert("mimeType".into(), json!(actual_mime));
        }
        let asset = Self { graph };
        asset.geometry()?;
        asset.animation_clips()?;
        Ok(asset)
    }
    /// Read-only retained document. Returned values use glTF coordinates/indices.
    pub fn document(&self) -> &Value {
        &self.graph.root
    }
    /// Evaluate the document's default pose, preserving UV0, in Anny Z-up meters.
    pub fn geometry(&self) -> Result<SurfaceMesh> {
        let mut mesh = self.graph.mesh()?;
        mesh.recalculate_normals()?;
        Ok(mesh)
    }
    /// Repackage into a self-contained GLB. Accessor/view indices are retained;
    /// multiple buffers are consolidated with aligned offsets. Images are embedded.
    pub fn to_glb(&self) -> Result<Vec<u8>> {
        let mut root = self.graph.root.clone();
        let mut bin = Vec::new();
        let mut offsets = Vec::new();
        for buffer in &self.graph.buffers {
            while !bin.len().is_multiple_of(4) {
                bin.push(0);
            }
            offsets.push(bin.len());
            ensure(
                buffer.len() <= LIMIT.saturating_sub(bin.len()),
                "packed GLB too large",
            )?;
            bin.extend_from_slice(buffer);
        }
        if let Some(views) = root.get_mut("bufferViews") {
            for view in views
                .as_array_mut()
                .ok_or_else(|| bad("invalid bufferViews"))?
            {
                let source = index(view, "buffer")?;
                let old = view
                    .get("byteOffset")
                    .map(|_| index(view, "byteOffset"))
                    .transpose()?
                    .unwrap_or(0);
                let offset = offsets
                    .get(source)
                    .ok_or_else(|| bad("invalid buffer reference"))?
                    .checked_add(old)
                    .ok_or_else(|| bad("offset overflow"))?;
                view["buffer"] = json!(0);
                view["byteOffset"] = json!(offset);
            }
        }
        root["buffers"] = json!([{"byteLength":bin.len()}]);
        let mut text = serde_json::to_vec(&root)?;
        while !text.len().is_multiple_of(4) {
            text.push(b' ');
        }
        while !bin.len().is_multiple_of(4) {
            bin.push(0);
        }
        let length = 28usize
            .checked_add(text.len())
            .and_then(|n| n.checked_add(bin.len()))
            .filter(|&n| n <= LIMIT)
            .ok_or_else(|| bad("GLB exceeds byte limit"))?;
        let mut out = Vec::with_capacity(length);
        out.extend_from_slice(b"glTF");
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&(length as u32).to_le_bytes());
        out.extend_from_slice(&(text.len() as u32).to_le_bytes());
        out.extend_from_slice(b"JSON");
        out.extend(text);
        out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        out.extend_from_slice(b"BIN\0");
        out.extend(bin);
        Ok(out)
    }
    pub fn save_glb(&self, path: impl AsRef<Path>) -> Result<()> {
        std::fs::write(path, self.to_glb()?)?;
        Ok(())
    }
}
pub(crate) fn array_mut<'a>(root: &'a mut Value, key: &str) -> Result<&'a mut Vec<Value>> {
    if root.get(key).is_none() {
        root[key] = json!([]);
    }
    root[key]
        .as_array_mut()
        .ok_or_else(|| bad(format!("glTF {key} must be an array")))
}
