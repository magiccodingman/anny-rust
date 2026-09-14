//! Geometry interchange for authoring/fitting. No subprocesses or Python.
//!
//! Reads OBJ (triangles/quads), ASCII/binary PLY, STL and glTF 2.0/GLB.
//! glTF import evaluates the selected scene's default morph weights and skin pose,
//! then converts Y-up meters to the model's Z-up meters. Animation sampling,
//! compressed geometry extensions and preservation of arbitrary materials are not
//! part of this geometry-only reader. Unsupported required extensions fail.
use crate::{ensure, math::*, scene::SurfaceMesh, Error, Result};
use base64::Engine;
use serde_json::Value;
use std::{
    collections::BTreeMap,
    io::{BufRead, Cursor, Write},
    path::{Component, Path},
};
const MAX_BYTES: usize = 512 * 1024 * 1024;
const MAX_VERTICES: usize = 10_000_000;
fn bad(s: impl Into<String>) -> Error {
    Error::Invalid(s.into())
}
fn read(path: &Path) -> Result<Vec<u8>> {
    ensure(
        std::fs::metadata(path)?.len() <= MAX_BYTES as u64,
        "mesh file exceeds 512 MiB limit",
    )?;
    Ok(std::fs::read(path)?)
}
pub fn load(path: impl AsRef<Path>) -> Result<SurfaceMesh> {
    let path = path.as_ref();
    let bytes = read(path)?;
    let ext = path
        .extension()
        .and_then(|x| x.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let mut mesh = match ext.as_str() {
        "obj" => obj(&bytes)?,
        "ply" => ply(&bytes)?,
        "stl" => stl(&bytes)?,
        "glb" | "gltf" => gltf(&bytes, Some(path.parent().unwrap_or(Path::new("."))))?,
        _ => return Err(bad("supported mesh inputs: OBJ, PLY, STL, glTF and GLB")),
    };
    mesh.validate()?;
    mesh.recalculate_normals()?;
    Ok(mesh)
}
/// In-memory parsing for hosts such as WASM. glTF buffers must be embedded.
pub fn from_bytes(bytes: &[u8], format: &str) -> Result<SurfaceMesh> {
    ensure(bytes.len() <= MAX_BYTES, "mesh input exceeds limit")?;
    let mut m = match format.to_ascii_lowercase().as_str() {
        "obj" => obj(bytes)?,
        "ply" => ply(bytes)?,
        "stl" => stl(bytes)?,
        "gltf" | "glb" => gltf(bytes, None)?,
        _ => return Err(bad("unknown mesh format")),
    };
    m.validate()?;
    m.recalculate_normals()?;
    Ok(m)
}
pub fn save(mesh: &SurfaceMesh, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    let ext = path
        .extension()
        .and_then(|x| x.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    std::fs::write(path, to_bytes(mesh, &ext)?)?;
    Ok(())
}
/// OBJ/PLY keep Z-up meters. STL has no unit/axis metadata; also written Z-up meters.
pub fn to_bytes(mesh: &SurfaceMesh, format: &str) -> Result<Vec<u8>> {
    mesh.validate()?;
    let mut out = Vec::new();
    match format.to_ascii_lowercase().as_str() {
        "obj" => {
            writeln!(out, "# anny-rust; meters, Z-up")?;
            for v in &mesh.positions {
                writeln!(out, "v {:.17} {:.17} {:.17}", v[0], v[1], v[2])?;
            }
            for u in &mesh.texcoords {
                writeln!(out, "vt {:.17} {:.17}", u[0], 1. - u[1])?;
            }
            for f in &mesh.triangles {
                if mesh.texcoords.is_empty() {
                    writeln!(out, "f {} {} {}", f[0] + 1, f[1] + 1, f[2] + 1)?;
                } else {
                    writeln!(
                        out,
                        "f {0}/{0} {1}/{1} {2}/{2}",
                        f[0] + 1,
                        f[1] + 1,
                        f[2] + 1
                    )?;
                }
            }
        }
        "ply" => {
            writeln!(out,"ply\nformat binary_little_endian 1.0\ncomment anny-rust meters Z-up\nelement vertex {}\nproperty double x\nproperty double y\nproperty double z\nelement face {}\nproperty list uchar uint vertex_indices\nend_header",mesh.positions.len(),mesh.triangles.len())?;
            for &x in mesh.positions.iter().flatten() {
                out.extend(x.to_le_bytes());
            }
            for f in &mesh.triangles {
                out.push(3);
                for &i in f {
                    out.extend(i.to_le_bytes());
                }
            }
        }
        "stl" => {
            let count =
                u32::try_from(mesh.triangles.len()).map_err(|_| bad("too many STL faces"))?;
            out.resize(80, 0);
            out[..9].copy_from_slice(b"anny-rust");
            out.extend(count.to_le_bytes());
            for f in &mesh.triangles {
                let [a, b, c] = f.map(|i| vec3(&mesh.positions[i as usize]));
                let n = (b - a)
                    .cross(&(c - a))
                    .try_normalize(1e-30)
                    .unwrap_or_else(Vec3::z);
                for v in [n, a, b, c] {
                    for x in v.iter() {
                        let x = *x as f32;
                        ensure(x.is_finite(), "STL coordinate exceeds f32 range")?;
                        out.extend(x.to_le_bytes());
                    }
                }
                out.extend([0, 0]);
            }
        }
        _ => {
            return Err(bad(
                "surface export supports OBJ/PLY/STL; use Scene for glTF/GLB",
            ))
        }
    }
    Ok(out)
}
fn triangulate(f: &[u32], v: &[[f64; 3]]) -> Result<Vec<[u32; 3]>> {
    ensure(
        (3..=4).contains(&f.len()),
        "geometry importer accepts triangles/quads, not arbitrary polygons",
    )?;
    ensure(
        f.iter().all(|&i| (i as usize) < v.len()),
        "mesh face index out of bounds",
    )?;
    if f.len() == 3 {
        return Ok(vec![[f[0], f[1], f[2]]]);
    }
    let a = (vec3(&v[f[0] as usize]) - vec3(&v[f[2] as usize])).norm_squared();
    let b = (vec3(&v[f[1] as usize]) - vec3(&v[f[3] as usize])).norm_squared();
    Ok(if a <= b {
        vec![[f[0], f[1], f[2]], [f[0], f[2], f[3]]]
    } else {
        vec![[f[0], f[1], f[3]], [f[1], f[2], f[3]]]
    })
}
fn obj(bytes: &[u8]) -> Result<SurfaceMesh> {
    let text = std::str::from_utf8(bytes).map_err(|_| bad("OBJ is not UTF-8"))?;
    let mut positions = Vec::new();
    let mut uv = Vec::new();
    let mut faces = Vec::new();
    let mut all_uv = true;
    fn id(s: &str, n: usize) -> Result<u32> {
        let x = s.parse::<i64>().map_err(|_| bad("invalid OBJ index"))?;
        let x = if x < 0 { n as i64 + x } else { x - 1 };
        ensure(x >= 0 && x < n as i64, "OBJ index outside vertex/UV array")?;
        Ok(x as u32)
    }
    for line in text.lines() {
        let p = line
            .split('#')
            .next()
            .unwrap_or("")
            .split_whitespace()
            .collect::<Vec<_>>();
        if p.is_empty() {
            continue;
        }
        match p[0] {
            "v" => {
                ensure(p.len() == 4, "OBJ vertex must have xyz")?;
                let mut v = [0.; 3];
                for i in 0..3 {
                    v[i] = p[i + 1]
                        .parse::<f64>()
                        .map_err(|_| bad("invalid OBJ number"))?;
                }
                positions.push(v);
                ensure(positions.len() <= MAX_VERTICES, "too many OBJ vertices")?;
            }
            "vt" => {
                ensure(p.len() >= 3, "OBJ UV needs two coordinates")?;
                uv.push([
                    p[1].parse::<f64>().map_err(|_| bad("invalid OBJ UV"))?,
                    1. - p[2].parse::<f64>().map_err(|_| bad("invalid OBJ UV"))?,
                ]);
            }
            "f" => {
                let mut f = Vec::new();
                for t in &p[1..] {
                    let t = t.split('/').collect::<Vec<_>>();
                    let vi = id(t[0], positions.len())?;
                    let ui = if t.len() > 1 && !t[1].is_empty() {
                        Some(id(t[1], uv.len())?)
                    } else {
                        all_uv = false;
                        None
                    };
                    f.push((vi, ui));
                }
                ensure(
                    (3..=4).contains(&f.len()),
                    "OBJ accepts triangle/quad faces",
                )?;
                faces.push(f);
            }
            _ => {} // o/g/usemtl do not terminate the geometry stream.
        }
    }
    let mut out = SurfaceMesh::default();
    let mut pairs = BTreeMap::new();
    if !all_uv || uv.is_empty() {
        out.positions = positions.clone();
        out.source_vertex_indices = (0..positions.len()).map(|x| x as u32).collect();
    }
    for face in faces {
        let mut f = Vec::new();
        for (v, u) in face {
            if !all_uv || uv.is_empty() {
                f.push(v);
                continue;
            }
            let u = u.ok_or_else(|| bad("partial OBJ UVs"))?;
            let idx = if let Some(&i) = pairs.get(&(v, u)) {
                i
            } else {
                let i = out.positions.len() as u32;
                pairs.insert((v, u), i);
                out.positions.push(positions[v as usize]);
                out.texcoords.push(uv[u as usize]);
                out.source_vertex_indices.push(v);
                i
            };
            f.push(idx);
        }
        out.triangles.extend(triangulate(&f, &out.positions)?);
    }
    if out.positions.is_empty() {
        out.positions = positions;
    }
    out.validate()?;
    Ok(out)
}
#[derive(Clone, Copy)]
enum Scalar {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}
impl Scalar {
    fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "char" | "int8" => Self::I8,
            "uchar" | "uint8" => Self::U8,
            "short" | "int16" => Self::I16,
            "ushort" | "uint16" => Self::U16,
            "int" | "int32" => Self::I32,
            "uint" | "uint32" => Self::U32,
            "float" | "float32" => Self::F32,
            "double" | "float64" => Self::F64,
            _ => return Err(bad("unsupported PLY scalar type")),
        })
    }
    fn size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }
    fn binary(self, b: &[u8], little: bool) -> f64 {
        macro_rules! value {
            ($t:ty) => {
                if little {
                    <$t>::from_le_bytes(b.try_into().unwrap()) as f64
                } else {
                    <$t>::from_be_bytes(b.try_into().unwrap()) as f64
                }
            };
        }
        match self {
            Self::I8 => b[0] as i8 as f64,
            Self::U8 => b[0] as f64,
            Self::I16 => value!(i16),
            Self::U16 => value!(u16),
            Self::I32 => value!(i32),
            Self::U32 => value!(u32),
            Self::F32 => value!(f32),
            Self::F64 => value!(f64),
        }
    }
}
fn integral(x: f64, bound: usize) -> Result<usize> {
    ensure(
        x.is_finite() && x >= 0. && x.fract() == 0. && x <= bound as f64,
        "invalid count or index",
    )?;
    Ok(x as usize)
}
fn ply(bytes: &[u8]) -> Result<SurfaceMesh> {
    #[derive(Default)]
    struct Element {
        name: String,
        count: usize,
        props: Vec<(String, Option<Scalar>, Scalar)>,
    }
    let mut cursor = Cursor::new(bytes);
    let mut line = String::new();
    cursor.read_line(&mut line)?;
    ensure(line.trim() == "ply", "bad PLY magic")?;
    let mut format = String::new();
    let mut elements: Vec<Element> = Vec::new();
    let mut ended = false;
    for _ in 0..4096 {
        line.clear();
        ensure(cursor.read_line(&mut line)? > 0, "truncated PLY header")?;
        let p = line.split_whitespace().collect::<Vec<_>>();
        if p.is_empty() {
            continue;
        }
        match p[0] {
            "format" => {
                ensure(p.len() == 3 && p[2] == "1.0", "unsupported PLY version")?;
                format = p[1].into();
            }
            "element" => {
                ensure(p.len() == 3, "bad PLY element")?;
                let count = p[2].parse::<usize>().map_err(|_| bad("bad PLY count"))?;
                ensure(count <= MAX_VERTICES * 4, "too many PLY elements")?;
                elements.push(Element {
                    name: p[1].into(),
                    count,
                    ..Default::default()
                });
            }
            "property" => {
                let e = elements
                    .last_mut()
                    .ok_or_else(|| bad("PLY property without element"))?;
                if p.get(1) == Some(&"list") {
                    ensure(p.len() == 5, "bad PLY list property")?;
                    e.props.push((
                        p[4].into(),
                        Some(Scalar::parse(p[2])?),
                        Scalar::parse(p[3])?,
                    ));
                } else {
                    ensure(p.len() == 3, "bad PLY property")?;
                    e.props.push((p[2].into(), None, Scalar::parse(p[1])?));
                }
            }
            "end_header" => {
                ended = true;
                break;
            }
            "comment" | "obj_info" => {}
            _ => return Err(bad("unknown PLY header declaration")),
        }
    }
    ensure(ended, "PLY header exceeds limit")?;
    ensure(
        ["ascii", "binary_little_endian", "binary_big_endian"].contains(&format.as_str()),
        "unsupported PLY encoding",
    )?;
    let body = &bytes[cursor.position() as usize..];
    let mut ascii = if format == "ascii" {
        std::str::from_utf8(body)
            .map_err(|_| bad("PLY text not UTF-8"))?
            .split_whitespace()
    } else {
        "".split_whitespace()
    };
    let mut offset = 0;
    let mut number = |ty: Scalar| -> Result<f64> {
        if format == "ascii" {
            ascii
                .next()
                .ok_or_else(|| bad("truncated PLY data"))?
                .parse()
                .map_err(|_| bad("invalid PLY number"))
        } else {
            let end = offset + ty.size();
            let b = body
                .get(offset..end)
                .ok_or_else(|| bad("truncated binary PLY"))?;
            offset = end;
            Ok(ty.binary(b, format == "binary_little_endian"))
        }
    };
    let mut mesh = SurfaceMesh::default();
    let mut faces = Vec::new();
    for e in elements {
        if e.name == "vertex" {
            for axis in ["x", "y", "z"] {
                ensure(
                    e.props.iter().any(|p| p.0 == axis && p.1.is_none()),
                    "PLY vertex is missing xyz",
                )?;
            }
        }
        for _ in 0..e.count {
            let mut v = [0.; 3];
            let mut f = None;
            for (name, count, ty) in &e.props {
                if let Some(ct) = count {
                    let n = integral(number(*ct)?, MAX_VERTICES)?;
                    let keep = e.name == "face"
                        && ["vertex_indices", "vertex_index"].contains(&name.as_str());
                    if keep {
                        ensure((3..=4).contains(&n), "PLY accepts triangle/quad faces")?;
                    }
                    let mut indices = Vec::new();
                    for _ in 0..n {
                        let x = number(*ty)?;
                        if keep {
                            indices.push(integral(x, u32::MAX as usize)? as u32);
                        }
                    }
                    if keep {
                        f = Some(indices);
                    }
                } else {
                    let x = number(*ty)?;
                    if e.name == "vertex" {
                        if let Some(i) = ["x", "y", "z"].iter().position(|a| a == name) {
                            v[i] = x;
                        }
                    }
                }
            }
            if e.name == "vertex" {
                ensure(mesh.positions.len() < MAX_VERTICES, "too many PLY vertices")?;
                mesh.positions.push(v);
            }
            if e.name == "face" {
                faces.push(f.ok_or_else(|| bad("PLY face missing vertex_indices"))?);
            }
        }
    }
    for f in faces {
        mesh.triangles.extend(triangulate(&f, &mesh.positions)?);
    }
    mesh.validate()?;
    Ok(mesh)
}
fn stl(bytes: &[u8]) -> Result<SurfaceMesh> {
    let count = if bytes.len() >= 84 {
        Some(u32::from_le_bytes(bytes[80..84].try_into().unwrap()) as usize)
    } else {
        None
    };
    let mut triangles: Vec<[[f64; 3]; 3]> = Vec::new();
    if count.is_some_and(|n| n.checked_mul(50).and_then(|x| x.checked_add(84)) == Some(bytes.len()))
    {
        let n = count.unwrap();
        ensure(n <= MAX_VERTICES, "too many STL faces")?;
        for f in bytes[84..].chunks_exact(50) {
            let mut tri = [[0.; 3]; 3];
            for (i, v) in tri.iter_mut().enumerate() {
                for (j, x) in v.iter_mut().enumerate() {
                    let off = 12 + (i * 3 + j) * 4;
                    *x = f32::from_le_bytes(f[off..off + 4].try_into().unwrap()) as f64;
                }
            }
            triangles.push(tri);
        }
    } else {
        let text = std::str::from_utf8(bytes).map_err(|_| bad("invalid binary/ASCII STL"))?;
        let mut v = Vec::new();
        for line in text.lines() {
            let p = line.split_whitespace().collect::<Vec<_>>();
            if p.first() == Some(&"vertex") {
                ensure(p.len() == 4, "bad STL vertex")?;
                let mut row = [0.; 3];
                for i in 0..3 {
                    row[i] = p[i + 1]
                        .parse()
                        .map_err(|_| bad("invalid STL coordinate"))?;
                }
                v.push(row);
            }
        }
        ensure(
            !v.is_empty() && v.len().is_multiple_of(3),
            "invalid STL triangle stream",
        )?;
        triangles.extend(v.chunks_exact(3).map(|x| [x[0], x[1], x[2]]));
    }
    let mut mesh = SurfaceMesh::default();
    let mut unique = BTreeMap::new();
    for t in triangles {
        let mut f = [0; 3];
        for (i, v) in t.iter().enumerate() {
            ensure(v.iter().all(|x| x.is_finite()), "nonfinite STL vertex")?;
            let key = v.map(|x| if x == 0. { 0 } else { x.to_bits() });
            f[i] = if let Some(&id) = unique.get(&key) {
                id
            } else {
                let id = mesh.positions.len() as u32;
                unique.insert(key, id);
                mesh.positions.push(*v);
                id
            };
        }
        mesh.triangles.push(f);
    }
    mesh.validate()?;
    Ok(mesh)
}
struct Gltf {
    root: Value,
    buffers: Vec<Vec<u8>>,
}
fn usize_field(v: &Value, key: &str) -> Result<usize> {
    v.get(key)
        .and_then(Value::as_u64)
        .and_then(|x| usize::try_from(x).ok())
        .ok_or_else(|| bad(format!("missing/invalid glTF {key}")))
}
fn optional_usize(v: &Value, key: &str, default: usize) -> Result<usize> {
    if v.get(key).is_none() {
        Ok(default)
    } else {
        usize_field(v, key)
    }
}
fn float_array<const N: usize>(v: Option<&Value>, default: [f64; N]) -> Result<[f64; N]> {
    let Some(v) = v else { return Ok(default) };
    let a = v
        .as_array()
        .ok_or_else(|| bad("glTF expected numeric array"))?;
    ensure(a.len() == N, "glTF array length mismatch")?;
    let mut o = [0.; N];
    for i in 0..N {
        o[i] = a[i]
            .as_f64()
            .filter(|x| x.is_finite())
            .ok_or_else(|| bad("invalid glTF number"))?;
    }
    Ok(o)
}
fn gltf(bytes: &[u8], directory: Option<&Path>) -> Result<SurfaceMesh> {
    let (root, bin) = if bytes.starts_with(b"glTF") {
        ensure(bytes.len() >= 20, "truncated GLB header")?;
        ensure(
            u32::from_le_bytes(bytes[4..8].try_into().unwrap()) == 2
                && u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize == bytes.len(),
            "GLB version/length mismatch",
        )?;
        let mut off = 12;
        let mut js = None;
        let mut bin = None;
        while off < bytes.len() {
            let h = bytes
                .get(off..off + 8)
                .ok_or_else(|| bad("truncated GLB chunk"))?;
            let n = u32::from_le_bytes(h[..4].try_into().unwrap()) as usize;
            ensure(n.is_multiple_of(4), "unaligned GLB chunk")?;
            let end = off
                .checked_add(8)
                .and_then(|x| x.checked_add(n))
                .ok_or_else(|| bad("GLB overflow"))?;
            let data = bytes
                .get(off + 8..end)
                .ok_or_else(|| bad("truncated GLB payload"))?;
            match &h[4..8] {
                b"JSON" => {
                    ensure(
                        js.is_none() && off == 12,
                        "GLB JSON must be first and unique",
                    )?;
                    js = Some(serde_json::from_slice(data)?);
                }
                b"BIN\0" => {
                    ensure(bin.is_none(), "multiple GLB binary chunks")?;
                    bin = Some(data.to_vec());
                }
                _ => {}
            }
            off = end;
        }
        (js.ok_or_else(|| bad("GLB missing JSON"))?, bin)
    } else {
        (serde_json::from_slice::<Value>(bytes)?, None)
    };
    ensure(
        root["asset"]["version"].as_str() == Some("2.0"),
        "only glTF 2.0 is supported",
    )?;
    ensure(
        root.get("extensionsRequired")
            .is_none_or(|v| v.as_array().is_some_and(Vec::is_empty)),
        "required glTF extension not supported by geometry reader",
    )?;
    let bs = root
        .get("buffers")
        .and_then(Value::as_array)
        .ok_or_else(|| bad("glTF missing buffers"))?;
    let mut buffers = Vec::new();
    let mut total = 0usize;
    for (i, b) in bs.iter().enumerate() {
        let data = if let Some(uri) = b.get("uri") {
            let uri = uri.as_str().ok_or_else(|| bad("invalid glTF buffer URI"))?;
            if uri.starts_with("data:") {
                let (header, text) = uri
                    .split_once(',')
                    .ok_or_else(|| bad("invalid buffer data URI"))?;
                ensure(
                    header.ends_with(";base64"),
                    "only base64 buffer data URIs supported",
                )?;
                base64::engine::general_purpose::STANDARD
                    .decode(text)
                    .map_err(|_| bad("invalid base64 buffer"))?
            } else {
                let dir = directory.ok_or_else(|| {
                    bad("external glTF buffer requires a filesystem path or embedded GLB")
                })?;
                ensure(
                    !uri.contains([':', '%', '\\', '?', '#']),
                    "external buffer URI must be a plain relative path",
                )?;
                let path = Path::new(uri);
                ensure(
                    path.components()
                        .all(|c| matches!(c, Component::Normal(_) | Component::CurDir)),
                    "external buffer path escapes document",
                )?;
                let root = dir.canonicalize()?;
                let path = root.join(path).canonicalize()?;
                ensure(
                    path.starts_with(root),
                    "external buffer symlink escapes document",
                )?;
                read(&path)?
            }
        } else {
            ensure(i == 0, "only buffer zero may use GLB binary chunk")?;
            bin.clone()
                .ok_or_else(|| bad("glTF buffer has no URI/BIN"))?
        };
        let n = usize_field(b, "byteLength")?;
        ensure(data.len() >= n, "glTF buffer shorter than declared")?;
        total = total
            .checked_add(data.len())
            .ok_or_else(|| bad("buffer size overflow"))?;
        ensure(total <= MAX_BYTES, "combined glTF buffers exceed limit")?;
        buffers.push(data[..n].to_vec());
    }
    Gltf { root, buffers }.mesh()
}
impl Gltf {
    fn array(&self, key: &str) -> Result<&Vec<Value>> {
        self.root
            .get(key)
            .and_then(Value::as_array)
            .ok_or_else(|| bad(format!("missing glTF {key}")))
    }
    fn view(&self, id: usize) -> Result<(&[u8], &Value)> {
        let v = self
            .array("bufferViews")?
            .get(id)
            .ok_or_else(|| bad("glTF bufferView out of range"))?;
        let b = self
            .buffers
            .get(usize_field(v, "buffer")?)
            .ok_or_else(|| bad("glTF buffer index out of range"))?;
        let off = optional_usize(v, "byteOffset", 0)?;
        let end = off
            .checked_add(usize_field(v, "byteLength")?)
            .ok_or_else(|| bad("buffer view overflow"))?;
        Ok((
            b.get(off..end)
                .ok_or_else(|| bad("glTF buffer view out of bounds"))?,
            v,
        ))
    }
    fn accessor(&self, id: usize, width: usize) -> Result<Vec<f64>> {
        let a = self
            .array("accessors")?
            .get(id)
            .ok_or_else(|| bad("glTF accessor out of range"))?;
        let w = match a["type"].as_str() {
            Some("SCALAR") => 1,
            Some("VEC2") => 2,
            Some("VEC3") => 3,
            Some("VEC4") => 4,
            Some("MAT4") => 16,
            _ => return Err(bad("unsupported glTF accessor type")),
        };
        ensure(w == width, "glTF attribute component count mismatch")?;
        let ty = match usize_field(a, "componentType")? {
            5120 => Scalar::I8,
            5121 => Scalar::U8,
            5122 => Scalar::I16,
            5123 => Scalar::U16,
            5125 => Scalar::U32,
            5126 => Scalar::F32,
            _ => return Err(bad("unsupported accessor component type")),
        };
        let count = usize_field(a, "count")?;
        ensure(
            count > 0 && count <= MAX_VERTICES && count * width <= MAX_BYTES / 8,
            "accessor count exceeds bounds",
        )?;
        let mut result = vec![0.; count * width];
        let normalized = a
            .get("normalized")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let decode = |b: &[u8]| {
            let n = ty.binary(b, true);
            if normalized {
                match ty {
                    Scalar::U8 => n / 255.,
                    Scalar::U16 => n / 65535.,
                    Scalar::I8 => (n / 127.).max(-1.),
                    Scalar::I16 => (n / 32767.).max(-1.),
                    _ => n,
                }
            } else {
                n
            }
        };
        if a.get("bufferView").is_some() {
            let (bytes, v) = self.view(usize_field(a, "bufferView")?)?;
            let start = optional_usize(a, "byteOffset", 0)?;
            let packed = width * ty.size();
            let stride = optional_usize(v, "byteStride", packed)?;
            ensure(
                stride >= packed && stride.is_multiple_of(ty.size()),
                "invalid glTF byte stride",
            )?;
            let end = start
                .checked_add(
                    (count - 1)
                        .checked_mul(stride)
                        .ok_or_else(|| bad("accessor overflow"))?,
                )
                .and_then(|x| x.checked_add(packed))
                .ok_or_else(|| bad("accessor overflow"))?;
            ensure(end <= bytes.len(), "glTF accessor out of bounds")?;
            for row in 0..count {
                for k in 0..width {
                    let off = start + row * stride + k * ty.size();
                    result[row * width + k] = decode(&bytes[off..off + ty.size()]);
                }
            }
        }
        if let Some(s) = a.get("sparse") {
            let n = usize_field(s, "count")?;
            ensure(n <= count, "sparse count exceeds accessor")?;
            let it = match usize_field(&s["indices"], "componentType")? {
                5121 => Scalar::U8,
                5123 => Scalar::U16,
                5125 => Scalar::U32,
                _ => return Err(bad("invalid sparse index type")),
            };
            let (ib, _) = self.view(usize_field(&s["indices"], "bufferView")?)?;
            let (vb, _) = self.view(usize_field(&s["values"], "bufferView")?)?;
            let io = optional_usize(&s["indices"], "byteOffset", 0)?;
            let vo = optional_usize(&s["values"], "byteOffset", 0)?;
            let mut prev = None;
            for row in 0..n {
                let off = io
                    .checked_add(row * it.size())
                    .ok_or_else(|| bad("sparse offset overflow"))?;
                let id = integral(
                    it.binary(
                        ib.get(
                            off..off
                                .checked_add(it.size())
                                .ok_or_else(|| bad("sparse index overflow"))?,
                        )
                        .ok_or_else(|| bad("sparse indices truncated"))?,
                        true,
                    ),
                    count - 1,
                )?;
                ensure(
                    prev.is_none_or(|p| id > p),
                    "sparse indices must be strictly increasing",
                )?;
                prev = Some(id);
                for k in 0..width {
                    let off = vo
                        .checked_add((row * width + k) * ty.size())
                        .ok_or_else(|| bad("sparse value overflow"))?;
                    result[id * width + k] = decode(
                        vb.get(
                            off..off
                                .checked_add(ty.size())
                                .ok_or_else(|| bad("sparse value overflow"))?,
                        )
                        .ok_or_else(|| bad("sparse values truncated"))?,
                    );
                }
            }
        }
        ensure(
            result.iter().all(|x| x.is_finite()),
            "nonfinite glTF attribute",
        )?;
        Ok(result)
    }
    fn mesh(&self) -> Result<SurfaceMesh> {
        let nodes = self.array("nodes")?;
        ensure(nodes.len() <= 100_000, "too many glTF nodes")?;
        let mut parents = vec![-1i32; nodes.len()];
        for (p, n) in nodes.iter().enumerate() {
            if let Some(c) = n.get("children") {
                for child in c.as_array().ok_or_else(|| bad("invalid node children"))? {
                    let i = child
                        .as_u64()
                        .and_then(|x| usize::try_from(x).ok())
                        .ok_or_else(|| bad("invalid child index"))?;
                    ensure(
                        i < nodes.len() && parents[i] < 0 && i != p,
                        "invalid/shared glTF child",
                    )?;
                    parents[i] = p as i32;
                }
            }
        }
        let order = propagation_order(&parents)?;
        let mut world = vec![Mat4::identity(); nodes.len()];
        for i in order {
            let n = &nodes[i];
            let local = if let Some(matrix) = n.get("matrix") {
                ensure(
                    ["translation", "rotation", "scale"]
                        .iter()
                        .all(|k| n.get(k).is_none()),
                    "node cannot combine matrix and TRS",
                )?;
                Mat4::from_column_slice(&float_array(Some(matrix), [0.; 16])?)
            } else {
                let t = float_array(n.get("translation"), [0.; 3])?;
                let q = float_array(n.get("rotation"), [0., 0., 0., 1.])?;
                let sc = float_array(n.get("scale"), [1.; 3])?;
                ensure(
                    (q.iter().map(|x| x * x).sum::<f64>() - 1.).abs() < 1e-4,
                    "invalid glTF quaternion",
                )?;
                let q = nalgebra::UnitQuaternion::new_normalize(nalgebra::Quaternion::new(
                    q[3], q[0], q[1], q[2],
                ));
                let mut m = rigid(q.to_rotation_matrix().matrix(), &vec3(&t));
                for col in 0..3 {
                    for row in 0..3 {
                        m[(row, col)] *= sc[col];
                    }
                }
                m
            };
            checked_rigid(&local)?;
            world[i] = if parents[i] < 0 {
                local
            } else {
                world[parents[i] as usize] * local
            };
        }
        let scene = optional_usize(&self.root, "scene", 0)?;
        let scenes = self.array("scenes")?;
        let scene = scenes
            .get(scene)
            .ok_or_else(|| bad("glTF default scene out of range"))?;
        let roots = scene
            .get("nodes")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("glTF scene missing roots"))?;
        let mut selected = vec![false; nodes.len()];
        let mut queue = Vec::new();
        for i in roots {
            let i = i
                .as_u64()
                .and_then(|x| usize::try_from(x).ok())
                .ok_or_else(|| bad("invalid scene root"))?;
            ensure(i < nodes.len() && parents[i] < 0, "invalid scene root")?;
            queue.push(i);
        }
        while let Some(i) = queue.pop() {
            ensure(!selected[i], "duplicate node in scene")?;
            selected[i] = true;
            if let Some(c) = nodes[i].get("children").and_then(Value::as_array) {
                for v in c {
                    queue.push(v.as_u64().unwrap() as usize);
                }
            }
        }
        let mut out = SurfaceMesh::default();
        let mesh_instances = nodes
            .iter()
            .enumerate()
            .filter(|(i, n)| selected[*i] && n.get("mesh").is_some())
            .count();
        for (i, node) in nodes.iter().enumerate() {
            if !selected[i] || node.get("mesh").is_none() {
                continue;
            }
            let mi = usize_field(node, "mesh")?;
            let mesh = self
                .array("meshes")?
                .get(mi)
                .ok_or_else(|| bad("mesh index out of range"))?;
            let primitives = mesh
                .get("primitives")
                .and_then(Value::as_array)
                .ok_or_else(|| bad("mesh has no primitives"))?;
            for primitive in primitives {
                ensure(
                    optional_usize(primitive, "mode", 4)? == 4,
                    "geometry reader supports triangle glTF primitives",
                )?;
                let attr = &primitive["attributes"];
                let mut pos = self.accessor(usize_field(attr, "POSITION")?, 3)?;
                let n = pos.len() / 3;
                if mesh_instances == 1 && primitives.len() == 1 {
                    if let Some(map) = mesh
                        .get("extras")
                        .and_then(|e| e.get("sourceVertexIndices"))
                    {
                        let ids: Vec<u32> = serde_json::from_value(map.clone())?;
                        ensure(
                            ids.len() == n && ids.iter().all(|&id| (id as usize) < MAX_VERTICES),
                            "invalid source vertex map",
                        )?;
                        out.source_vertex_indices = ids;
                    }
                }
                if let Some(targets) = primitive.get("targets").and_then(Value::as_array) {
                    let weights = node
                        .get("weights")
                        .or_else(|| mesh.get("weights"))
                        .and_then(Value::as_array);
                    if let Some(w) = weights {
                        ensure(w.len() == targets.len(), "morph weight count mismatch")?;
                    }
                    for (t, target) in targets.iter().enumerate() {
                        let weight = if let Some(w) = weights {
                            w[t].as_f64()
                                .filter(|x| x.is_finite())
                                .ok_or_else(|| bad("invalid morph weight"))?
                        } else {
                            0.
                        };
                        if target.get("POSITION").is_some() {
                            let delta = self.accessor(usize_field(target, "POSITION")?, 3)?;
                            ensure(delta.len() == pos.len(), "morph position count mismatch")?;
                            for (p, d) in pos.iter_mut().zip(delta) {
                                *p += weight * d;
                            }
                        }
                    }
                }
                let mut transformed = Vec::with_capacity(n);
                if node.get("skin").is_some() {
                    let skin = self
                        .array("skins")?
                        .get(usize_field(node, "skin")?)
                        .ok_or_else(|| bad("skin index out of range"))?;
                    let joints = skin
                        .get("joints")
                        .and_then(Value::as_array)
                        .ok_or_else(|| bad("skin missing joints"))?;
                    ensure(!joints.is_empty(), "skin has no joints")?;
                    let ib = if skin.get("inverseBindMatrices").is_some() {
                        self.accessor(usize_field(skin, "inverseBindMatrices")?, 16)?
                    } else {
                        (0..joints.len())
                            .flat_map(|_| Mat4::identity().as_slice().to_vec())
                            .collect()
                    };
                    ensure(ib.len() == joints.len() * 16, "bind matrix count mismatch")?;
                    let mut transforms = Vec::new();
                    for (k, j) in joints.iter().enumerate() {
                        let j = j
                            .as_u64()
                            .and_then(|x| usize::try_from(x).ok())
                            .ok_or_else(|| bad("invalid joint node"))?;
                        ensure(
                            j < world.len() && selected[j],
                            "skin joint outside selected scene",
                        )?;
                        transforms
                            .push(world[j] * Mat4::from_column_slice(&ib[k * 16..(k + 1) * 16]));
                    }
                    let mut sets = Vec::new();
                    for set in 0..64 {
                        let key = format!("JOINTS_{set}");
                        if attr.get(&key).is_none() {
                            break;
                        }
                        let ids = self.accessor(usize_field(attr, &key)?, 4)?;
                        let ws = self.accessor(usize_field(attr, &format!("WEIGHTS_{set}"))?, 4)?;
                        ensure(
                            ids.len() == n * 4 && ws.len() == n * 4,
                            "skinning attribute length mismatch",
                        )?;
                        sets.push((ids, ws));
                    }
                    ensure(!sets.is_empty(), "skinned primitive has no weights")?;
                    for key in attr
                        .as_object()
                        .ok_or_else(|| bad("invalid primitive attributes"))?
                        .keys()
                    {
                        if let Some(s) = key
                            .strip_prefix("JOINTS_")
                            .or_else(|| key.strip_prefix("WEIGHTS_"))
                        {
                            let set = s
                                .parse::<usize>()
                                .map_err(|_| bad("invalid skin attribute set"))?;
                            ensure(
                                set < sets.len(),
                                "skin attribute sets must be paired and consecutive (maximum 64)",
                            )?;
                        }
                    }
                    for v in 0..n {
                        let p = vec3(&pos[v * 3..v * 3 + 3]);
                        let mut result = Vec3::zeros();
                        let mut sum = 0.;
                        for (ids, ws) in &sets {
                            for s in 0..4 {
                                let id =
                                    integral(ids[v * 4 + s], transforms.len().saturating_sub(1))?;
                                let w = ws[v * 4 + s];
                                ensure(w >= 0., "negative skin weight")?;
                                sum += w;
                                result += w * point(&transforms[id], &p);
                            }
                        }
                        ensure((sum - 1.).abs() < 1e-4, "skin weights must sum to one")?;
                        transformed.push(result);
                    }
                } else {
                    transformed.extend(pos.chunks_exact(3).map(|p| point(&world[i], &vec3(p))));
                }
                let offset =
                    u32::try_from(out.positions.len()).map_err(|_| bad("merged mesh too large"))?;
                ensure(
                    out.positions.len() + n <= MAX_VERTICES,
                    "merged mesh exceeds vertex limit",
                )?;
                // Inverse of the export root rotation: glTF Y-up -> Anny Z-up.
                out.positions
                    .extend(transformed.iter().map(|p| [p.x, -p.z, p.y]));
                let ids = if primitive.get("indices").is_some() {
                    self.accessor(usize_field(primitive, "indices")?, 1)?
                } else {
                    (0..n).map(|i| i as f64).collect()
                };
                ensure(
                    ids.len().is_multiple_of(3),
                    "triangle index count not divisible by three",
                )?;
                for f in ids.chunks_exact(3) {
                    out.triangles.push([
                        offset + integral(f[0], n - 1)? as u32,
                        offset + integral(f[1], n - 1)? as u32,
                        offset + integral(f[2], n - 1)? as u32,
                    ]);
                }
            }
        }
        out.validate()?;
        Ok(out)
    }
}
