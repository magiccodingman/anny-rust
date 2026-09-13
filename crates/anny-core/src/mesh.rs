// Port of NAVER Anny mesh operations, Copyright (C) 2025 NAVER Corp.
// SPDX-License-Identifier: Apache-2.0
use crate::{ensure, math::*, model::ModelData, tensor::Kind, Error, Result, Tensor};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{BufRead, BufReader, Write},
    path::Path,
};

#[derive(Clone, Debug, Default)]
pub struct ObjGroup {
    pub faces: Vec<Vec<usize>>,
    pub uv_faces: Vec<Vec<usize>>,
}
#[derive(Clone, Debug, Default)]
pub struct Obj {
    pub vertices: Vec<Vec3>,
    pub uv: Vec<[f64; 2]>,
    pub groups: BTreeMap<String, ObjGroup>,
}
pub fn load_obj(path: impl AsRef<Path>) -> Result<Obj> {
    let file = std::fs::File::open(path.as_ref())?;
    let mut obj = Obj::default();
    let mut group = "noname".to_string();
    for (line_id, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        let parts: Vec<_> = line.split_whitespace().collect();
        if parts.is_empty() || parts[0].starts_with('#') {
            continue;
        }
        let number = |s: &str| {
            s.parse::<f64>()
                .map_err(|_| Error::Invalid(format!("OBJ line {} invalid number {s}", line_id + 1)))
        };
        let index = |s: &str, bound: usize| -> Result<usize> {
            let i = s
                .parse::<i64>()
                .map_err(|_| Error::Invalid("invalid OBJ index".into()))?;
            let i = if i < 0 { bound as i64 + i } else { i - 1 };
            ensure(i >= 0 && i < (bound as i64), "OBJ index out of range")?;
            Ok(i as usize)
        };
        match parts[0] {
            "o" if !obj.vertices.is_empty() => break,
            "v" => {
                ensure(parts.len() == 4, "OBJ vertices need three coordinates")?;
                obj.vertices.push(Vec3::new(
                    number(parts[1])?,
                    number(parts[2])?,
                    number(parts[3])?,
                ));
            }
            "vt" => {
                ensure(parts.len() >= 3, "OBJ UV needs two coordinates")?;
                obj.uv.push([number(parts[1])?, number(parts[2])?]);
            }
            "g" => {
                ensure(parts.len() > 1, "unnamed OBJ group")?;
                group = parts[1].into();
            }
            "f" => {
                ensure(
                    parts.len() == 4 || parts.len() == 5,
                    "only triangle/quad OBJ faces are supported",
                )?;
                let (mut face, mut uv) = (Vec::new(), Vec::new());
                for token in &parts[1..] {
                    let parts: Vec<_> = token.split('/').collect();
                    face.push(index(parts[0], obj.vertices.len())?);
                    if parts.len() > 1 && !parts[1].is_empty() {
                        uv.push(index(parts[1], obj.uv.len())?);
                    }
                }
                ensure(
                    uv.is_empty() || uv.len() == face.len(),
                    "partial OBJ UV face",
                )?;
                let g = obj.groups.entry(group.clone()).or_default();
                g.faces.push(face);
                g.uv_faces.push(uv);
            }
            _ => {}
        }
    }
    ensure(
        !obj.vertices.is_empty() && obj.vertices.iter().all(|v| v.iter().all(|x| x.is_finite())),
        "empty/non-finite OBJ",
    )?;
    Ok(obj)
}
pub fn pack_faces(faces: &[Vec<usize>]) -> Result<Tensor> {
    let k = faces.first().map_or(3, Vec::len);
    ensure(
        faces.iter().all(|f| f.len() == k),
        "mixed face sizes must be triangulated first",
    )?;
    Ok(Tensor::indices(
        vec![faces.len(), k],
        faces.iter().flatten().copied().collect(),
    ))
}
pub fn save_obj(
    path: impl AsRef<Path>,
    vertices: &Tensor,
    faces: &Tensor,
    uv: Option<(&Tensor, &Tensor)>,
) -> Result<()> {
    ensure(
        vertices.shape.len() == 2 && vertices.shape[1] == 3,
        "OBJ export needs [V,3] vertices",
    )?;
    faces.checked_indices(vertices.shape[0], "OBJ faces")?;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    for v in vertices.data.chunks_exact(3) {
        writeln!(f, "v {:.17} {:.17} {:.17}", v[0], v[1], v[2])?;
    }
    if let Some((t, ft)) = uv {
        t.expect_shape(&[t.shape[0], 2], "UV")?;
        ft.expect_shape(&faces.shape, "UV faces")?;
        ft.checked_indices(t.shape[0], "UV faces")?;
        for p in t.data.chunks_exact(2) {
            writeln!(f, "vt {:.17} {:.17}", p[0], p[1])?;
        }
    }
    for (i, face) in faces.data.chunks_exact(faces.shape[1]).enumerate() {
        write!(f, "f")?;
        for (s, &v) in face.iter().enumerate() {
            if let Some((_, ft)) = uv {
                write!(
                    f,
                    " {}/{}",
                    v as usize + 1,
                    ft.data[i * face.len() + s] as usize + 1
                )?;
            } else {
                write!(f, " {}", v as usize + 1)?;
            }
        }
        writeln!(f)?;
    }
    f.flush()?;
    Ok(())
}
pub fn triangulate(d: &mut ModelData) -> Result<()> {
    let faces = d.get("faces")?;
    if faces.shape[1] == 3 {
        return Ok(());
    }
    let v = d.get("template_vertices")?;
    let mut indices = Vec::new();
    for (i, f) in faces.data.chunks_exact(4).enumerate() {
        let a = vec3(&v.data[f[0] as usize * 3..f[0] as usize * 3 + 3]);
        let b = vec3(&v.data[f[1] as usize * 3..f[1] as usize * 3 + 3]);
        let c = vec3(&v.data[f[2] as usize * 3..f[2] as usize * 3 + 3]);
        let e = vec3(&v.data[f[3] as usize * 3..f[3] as usize * 3 + 3]);
        let ids = if (a - c).norm() < (b - e).norm() {
            [0, 1, 2, 2, 3, 0]
        } else {
            [0, 1, 3, 3, 1, 2]
        };
        indices.extend(ids.map(|k| i * 4 + k));
    }
    for name in ["faces", "face_texture_coordinate_indices"] {
        if let Some(t) = d.arrays.get(name) {
            let data = indices.iter().map(|&i| t.data[i]).collect();
            d.put(
                name,
                Tensor {
                    shape: vec![indices.len() / 3, 3],
                    data,
                    kind: Kind::Index,
                },
            );
        }
    }
    Ok(())
}
pub fn filter_faces(d: &mut ModelData, keep: &[usize]) -> Result<()> {
    for name in ["faces", "face_texture_coordinate_indices"] {
        if let Some(t) = d.arrays.get(name) {
            let new = t.select(0, keep)?;
            d.put(name, new);
        }
    }
    Ok(())
}
pub fn remove_unattached_vertices(d: &mut ModelData) -> Result<()> {
    let n = d.vertex_count();
    let keep: Vec<_> = d
        .get("faces")?
        .checked_indices(n, "faces")?
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    ensure(
        !keep.is_empty(),
        "cannot compact a mesh with no attached vertices",
    )?;
    let mut remap = vec![0; n];
    for (i, &id) in keep.iter().enumerate() {
        remap[id] = i;
    }
    for (name, axis) in [
        ("template_vertices", 0),
        ("blendshapes", 1),
        ("vertex_bone_weights", 0),
        ("vertex_bone_indices", 0),
        ("base_mesh_vertex_indices", 0),
    ] {
        let t = d.get(name)?.select(axis, &keep)?;
        d.put(name, t);
    }
    let f = d.arrays.get_mut("faces").unwrap();
    for id in &mut f.data {
        *id = remap[*id as usize] as f64;
    }
    Ok(())
}
pub fn edit_mesh(d: &mut ModelData) -> Result<()> {
    let faces = d.get("faces")?;
    let uv = d.get("face_texture_coordinate_indices")?;
    ensure(faces.shape[1] == 4, "mesh edits require original quad mesh")?;
    let discard = |i: usize| (1778..1794).contains(&i) || (8450..8466).contains(&i);
    let (mut fs, mut ts, mut uvmap) = (Vec::new(), Vec::new(), BTreeMap::new());
    for (face, t) in faces.data.chunks_exact(4).zip(uv.data.chunks_exact(4)) {
        if face.iter().any(|&id| discard(id as usize)) {
            for (&v, &u) in face.iter().zip(t) {
                if let Some(prev) = uvmap.insert(v as usize, u as usize) {
                    ensure(prev == u as usize, "edited cap has inconsistent UV seam")?;
                }
            }
        } else {
            fs.extend_from_slice(face);
            ts.extend_from_slice(t);
        }
    }
    let caps = [
        [8437, 8438, 8439, 8440],
        [8436, 8437, 8440, 8441],
        [8435, 8436, 8441, 8442],
        [8434, 8435, 8442, 8443],
        [8449, 8434, 8443, 8444],
        [8448, 8449, 8444, 8445],
        [8447, 8448, 8445, 8446],
        [1762, 1771, 1770, 1763],
        [1763, 1770, 1769, 1764],
        [1764, 1769, 1768, 1765],
        [1765, 1768, 1767, 1766],
        [1762, 1777, 1772, 1771],
        [1777, 1776, 1773, 1772],
        [1776, 1775, 1774, 1773],
    ];
    for face in caps {
        for v in face {
            fs.push(v as f64);
            ts.push(
                *uvmap
                    .get(&v)
                    .ok_or_else(|| Error::Invalid("missing cap UV".into()))? as f64,
            );
        }
    }
    let shape = vec![fs.len() / 4, 4];
    d.put(
        "faces",
        Tensor {
            shape: shape.clone(),
            data: fs,
            kind: Kind::Index,
        },
    );
    d.put(
        "face_texture_coordinate_indices",
        Tensor {
            shape,
            data: ts,
            kind: Kind::Index,
        },
    );
    Ok(())
}
pub fn compact_skinning_weights(d: &mut ModelData) -> Result<()> {
    loop {
        let w = d.get("vertex_bone_weights")?;
        let ids = d.get("vertex_bone_indices")?;
        let k = w.shape[1];
        if k <= 1 {
            break;
        }
        let mins: Vec<_> = w
            .data
            .chunks_exact(k)
            .map(|r| (0..k).min_by(|&a, &b| r[a].total_cmp(&r[b])).unwrap())
            .collect();
        if mins
            .iter()
            .enumerate()
            .any(|(v, &i)| w.data[v * k + i] > 0.)
        {
            break;
        }
        let (mut ws, mut is) = (Vec::new(), Vec::new());
        for (v, &m) in mins.iter().enumerate() {
            for s in 0..k {
                if s != m {
                    ws.push(w.data[v * k + s]);
                    is.push(ids.data[v * k + s]);
                }
            }
        }
        let shape = vec![w.shape[0], k - 1];
        d.put(
            "vertex_bone_weights",
            Tensor {
                shape: shape.clone(),
                data: ws,
                kind: Kind::Float,
            },
        );
        d.put(
            "vertex_bone_indices",
            Tensor {
                shape,
                data: is,
                kind: Kind::Index,
            },
        );
    }
    Ok(())
}
/// Unique edges and incidence counts. A count of one denotes a boundary edge.
pub fn edges(faces: &Tensor) -> Result<BTreeMap<(usize, usize), usize>> {
    ensure(
        faces.shape.len() == 2 && faces.shape[1] >= 3,
        "invalid faces",
    )?;
    let mut out = BTreeMap::new();
    for f in faces.data.chunks_exact(faces.shape[1]) {
        for i in 0..f.len() {
            let (a, b) = (f[i] as usize, f[(i + 1) % f.len()] as usize);
            *out.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    Ok(out)
}
/// Closest point on a triangle, with barycentrics in vertex order.
pub fn closest_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> (Vec3, [f64; 3]) {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(&ap);
    let d2 = ac.dot(&ap);
    if d1 <= 0. && d2 <= 0. {
        return (a, [1., 0., 0.]);
    }
    let bp = p - b;
    let d3 = ab.dot(&bp);
    let d4 = ac.dot(&bp);
    if d3 >= 0. && d4 <= d3 {
        return (b, [0., 1., 0.]);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0. && d1 >= 0. && d3 <= 0. {
        let v = d1 / (d1 - d3);
        return (a + v * ab, [1. - v, v, 0.]);
    }
    let cp = p - c;
    let d5 = ab.dot(&cp);
    let d6 = ac.dot(&cp);
    if d6 >= 0. && d5 <= d6 {
        return (c, [0., 0., 1.]);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0. && d2 >= 0. && d6 <= 0. {
        let w = d2 / (d2 - d6);
        return (a + w * ac, [1. - w, 0., w]);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0. && (d4 - d3) >= 0. && (d5 - d6) >= 0. {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (b + w * (c - b), [0., 1. - w, w]);
    }
    let sum = va + vb + vc;
    if sum.abs() < 1e-30 {
        let mut best = (a, [1., 0., 0.]);
        let mut dist = (p - a).norm_squared();
        for (x, y, i, j) in [(a, b, 0, 1), (b, c, 1, 2), (c, a, 2, 0)] {
            let delta = y - x;
            let t = if delta.norm_squared() == 0. {
                0.
            } else {
                ((p - x).dot(&delta) / delta.norm_squared()).clamp(0., 1.)
            };
            let q = x + t * delta;
            let dd = (p - q).norm_squared();
            if dd < dist {
                let mut w = [0.; 3];
                w[i] = 1. - t;
                w[j] = t;
                best = (q, w);
                dist = dd;
            }
        }
        return best;
    }
    let v = vb / sum;
    let w = vc / sum;
    (a + v * ab + w * ac, [1. - v - w, v, w])
}
// Float32 projection preserves Warp 1.9's arithmetic/region conventions.
// Adapted from NVIDIA Warp intersect.h / mesh.h, Copyright (c) 2022
// NVIDIA CORPORATION & AFFILIATES. SPDX-License-Identifier: Apache-2.0
fn closest_triangle_f32(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Option<(f64, [f64; 3])> {
    type V = [f32; 3];
    fn sub(a: V, b: V) -> V {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }
    fn dot(a: V, b: V) -> f32 {
        (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]
    }
    let cast = |v: Vec3| [v[0] as f32, v[1] as f32, v[2] as f32];
    let (p, a, b, c) = (cast(p), cast(a), cast(b), cast(c));
    let ab = sub(b, a);
    let ac = sub(c, a);
    let bc = sub(c, b);
    let normal = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    if dot(normal, normal).sqrt() / (dot(ab, ab) + dot(ac, ac) + dot(bc, bc)) < 1e-6 {
        return None;
    }
    let ap = sub(p, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    let (v, w) = if d1 <= 0. && d2 <= 0. {
        (0., 0.)
    } else {
        let bp = sub(p, b);
        let d3 = dot(ab, bp);
        let d4 = dot(ac, bp);
        if d3 >= 0. && d4 <= d3 {
            (1., 0.)
        } else {
            let vc = d1 * d4 - d3 * d2;
            if vc <= 0. && d1 >= 0. && d3 <= 0. {
                (d1 / (d1 - d3), 0.)
            } else {
                let cp = sub(p, c);
                let d5 = dot(ab, cp);
                let d6 = dot(ac, cp);
                if d6 >= 0. && d5 <= d6 {
                    (0., 1.)
                } else {
                    let vb = d5 * d2 - d1 * d6;
                    if vb <= 0. && d2 >= 0. && d6 <= 0. {
                        (0., d2 / (d2 - d6))
                    } else {
                        let va = d3 * d6 - d5 * d4;
                        if va <= 0. && (d4 - d3) >= 0. && (d5 - d6) >= 0. {
                            let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                            (1. - w, w)
                        } else {
                            let denom = 1. / (va + vb + vc);
                            (vb * denom, vc * denom)
                        }
                    }
                }
            }
        }
    };
    let u = 1. - v - w;
    let w = 1. - u - v;
    let q = [
        u * a[0] + v * b[0] + w * c[0],
        u * a[1] + v * b[1] + w * c[1],
        u * a[2] + v * b[2] + w * c[2],
    ];
    let delta = sub(q, p);
    let dist = dot(delta, delta);
    let u = 1. - v - w;
    if !dist.is_finite() {
        return None;
    }
    Some((dist as f64, [u as f64, v as f64, 1. - u as f64 - v as f64]))
}
#[derive(Clone, Debug)]
struct BvhNode {
    lo: Vec3,
    hi: Vec3,
    left: usize,
    right: usize,
    start: usize,
    end: usize,
}
/// Deterministic CPU triangle BVH used by projection and collision queries.
#[derive(Clone, Debug)]
pub struct MeshBvh {
    vertices: Vec<Vec3>,
    faces: Vec<[usize; 3]>,
    order: Vec<usize>,
    nodes: Vec<BvhNode>,
}
impl MeshBvh {
    pub fn new(vertices: &Tensor, faces: &Tensor) -> Result<Self> {
        ensure(
            vertices.shape.len() == 2
                && vertices.shape[1] == 3
                && faces.shape.len() == 2
                && faces.shape[1] == 3
                && faces.shape[0] > 0,
            "BVH needs vertices and nonempty triangular faces",
        )?;
        vertices.validate()?;
        let ids = faces.checked_indices(vertices.shape[0], "BVH faces")?;
        let mut b = Self {
            vertices: vertices.data.chunks_exact(3).map(vec3).collect(),
            faces: ids.chunks_exact(3).map(|x| [x[0], x[1], x[2]]).collect(),
            order: (0..faces.shape[0]).collect(),
            nodes: Vec::new(),
        };
        b.build(0, b.order.len());
        Ok(b)
    }
    /// Warp-compatible CPU SAH construction for projection tie ordering.
    /// Adapted from NVIDIA Warp bvh.cpp (Apache-2.0), Copyright (c) 2022 NVIDIA.
    pub fn new_projection(vertices: &Tensor, faces: &Tensor) -> Result<Self> {
        let mut b = Self::new(vertices, faces)?;
        b.order = (0..b.faces.len()).collect();
        b.nodes.clear();
        let primitive: Vec<_> = b
            .faces
            .iter()
            .map(|f| {
                let mut lo = [f32::MAX; 3];
                let mut hi = [-f32::MAX; 3];
                for &v in f {
                    for a in 0..3 {
                        lo[a] = lo[a].min(b.vertices[v][a] as f32);
                        hi[a] = hi[a].max(b.vertices[v][a] as f32);
                    }
                }
                (lo, hi)
            })
            .collect();
        b.build_sah(0, b.order.len(), 0, &primitive);
        Ok(b)
    }
    fn build_sah(
        &mut self,
        start: usize,
        end: usize,
        depth: usize,
        primitive: &[([f32; 3], [f32; 3])],
    ) -> usize {
        fn merge(a: ([f32; 3], [f32; 3]), b: ([f32; 3], [f32; 3])) -> ([f32; 3], [f32; 3]) {
            (
                std::array::from_fn(|i| a.0[i].min(b.0[i])),
                std::array::from_fn(|i| a.1[i].max(b.1[i])),
            )
        }
        fn area(b: ([f32; 3], [f32; 3])) -> f32 {
            let e: [f32; 3] = std::array::from_fn(|i| b.1[i] - b.0[i]);
            2. * (e[0] * e[1] + e[0] * e[2] + e[1] * e[2])
        }
        let empty = ([f32::MAX; 3], [-f32::MAX; 3]);
        let mut bounds = empty;
        for &f in &self.order[start..end] {
            bounds = merge(bounds, primitive[f]);
        }
        let id = self.nodes.len();
        self.nodes.push(BvhNode {
            lo: Vec3::new(bounds.0[0] as f64, bounds.0[1] as f64, bounds.0[2] as f64),
            hi: Vec3::new(bounds.1[0] as f64, bounds.1[1] as f64, bounds.1[2] as f64),
            left: usize::MAX,
            right: usize::MAX,
            start,
            end,
        });
        if end - start <= 4 || depth >= 31 {
            return id;
        }
        let extent: [f32; 3] = std::array::from_fn(|i| bounds.1[i] - bounds.0[i]);
        let mut axis = 0;
        for i in 1..3 {
            if extent[i].abs() > extent[axis].abs() {
                axis = i;
            }
        }
        let lo = bounds.0[axis];
        let range = bounds.1[axis] - lo;
        let mut buckets = [empty; 16];
        let mut counts = [0usize; 16];
        let center = |f: usize| (primitive[f].0[axis] + primitive[f].1[axis]) * 0.5;
        for &f in &self.order[start..end] {
            let bucket = ((16. * (center(f) - lo) / range) as usize).min(15);
            buckets[bucket] = merge(buckets[bucket], primitive[f]);
            counts[bucket] += 1;
        }
        let mut left = [0.; 15];
        let mut right = [0.; 15];
        let mut nl = [0usize; 15];
        let mut nr = [0usize; 15];
        let (mut lb, mut rb) = (empty, empty);
        let (mut lc, mut rc) = (0, 0);
        for i in 0..15 {
            lb = merge(lb, buckets[i]);
            rb = merge(rb, buckets[15 - i]);
            left[i] = area(lb);
            right[14 - i] = area(rb);
            lc += counts[i];
            rc += counts[15 - i];
            nl[i] = lc;
            nr[14 - i] = rc;
        }
        let inv = 1. / area(bounds);
        let mut best = f32::MAX;
        let mut split = 0;
        for i in 0..15 {
            let cost = (left[i] * inv) * (nl[i] as f32) + (right[i] * inv) * (nr[i] as f32);
            if cost < best {
                best = cost;
                split = i;
            }
        }
        let cut = lo + (split + 1) as f32 * range / 16.;
        // Bidirectional partition matches std::partition's pointer-range ordering.
        let (mut first, mut last) = (start, end);
        loop {
            while first < last && center(self.order[first]) < cut {
                first += 1;
            }
            if first == last {
                break;
            }
            loop {
                last -= 1;
                if first == last || center(self.order[last]) < cut {
                    break;
                }
            }
            if first == last {
                break;
            }
            self.order.swap(first, last);
            first += 1;
        }
        let mid = if first == start || first == end {
            (start + end) / 2
        } else {
            first
        };
        let left = self.build_sah(start, mid, depth + 1, primitive);
        let right = self.build_sah(mid, end, depth + 1, primitive);
        self.nodes[id].left = left;
        self.nodes[id].right = right;
        id
    }
    fn bounds(&self, start: usize, end: usize) -> (Vec3, Vec3) {
        let mut lo = Vec3::repeat(f64::INFINITY);
        let mut hi = Vec3::repeat(f64::NEG_INFINITY);
        for &f in &self.order[start..end] {
            for &v in &self.faces[f] {
                for k in 0..3 {
                    lo[k] = lo[k].min(self.vertices[v][k]);
                    hi[k] = hi[k].max(self.vertices[v][k]);
                }
            }
        }
        (lo, hi)
    }
    fn build(&mut self, start: usize, end: usize) -> usize {
        let (lo, hi) = self.bounds(start, end);
        let id = self.nodes.len();
        self.nodes.push(BvhNode {
            lo,
            hi,
            left: usize::MAX,
            right: usize::MAX,
            start,
            end,
        });
        if end - start > 8 {
            let extent = hi - lo;
            let axis = (0..3)
                .max_by(|&a, &b| extent[a].total_cmp(&extent[b]))
                .unwrap();
            let verts = &self.vertices;
            let faces = &self.faces;
            self.order[start..end].sort_unstable_by(|&a, &b| {
                let center = |f: usize| faces[f].iter().map(|&i| verts[i][axis]).sum::<f64>();
                center(a).total_cmp(&center(b)).then(a.cmp(&b))
            });
            let mid = (start + end) / 2;
            let l = self.build(start, mid);
            let r = self.build(mid, end);
            self.nodes[id].left = l;
            self.nodes[id].right = r;
        }
        id
    }
    fn box_distance(node: &BvhNode, p: Vec3) -> f64 {
        (0..3)
            .map(|i| {
                let q = if p[i] < node.lo[i] {
                    node.lo[i] - p[i]
                } else if p[i] > node.hi[i] {
                    p[i] - node.hi[i]
                } else {
                    0.
                };
                q * q
            })
            .sum()
    }
    pub fn closest(&self, p: Vec3) -> (f64, usize, [f64; 3]) {
        self.closest_impl(p, false)
    }
    pub fn closest_portable_f32(&self, p: Vec3) -> (f64, usize, [f64; 3]) {
        self.closest_impl(p, true)
    }
    fn closest_impl(&self, p: Vec3, float32: bool) -> (f64, usize, [f64; 3]) {
        let distance = |node: &BvhNode| {
            if float32 {
                let q: [f32; 3] = std::array::from_fn(|i| {
                    let v = p[i] as f32;
                    let c = v.max(node.lo[i] as f32).min(node.hi[i] as f32);
                    v - c
                });
                ((q[0] * q[0] + q[1] * q[1]) + q[2] * q[2]) as f64
            } else {
                Self::box_distance(node, p)
            }
        };
        let mut stack = vec![0];
        let mut best = f64::INFINITY;
        let mut face = usize::MAX;
        let mut bary = [0.; 3];
        while let Some(id) = stack.pop() {
            let node = &self.nodes[id];
            if distance(node) > best {
                continue;
            }
            if node.left == usize::MAX {
                for &f in &self.order[node.start..node.end] {
                    let [a, b, c] = self.faces[f];
                    let (dist, w) = if float32 {
                        let Some(pair) = closest_triangle_f32(
                            p,
                            self.vertices[a],
                            self.vertices[b],
                            self.vertices[c],
                        ) else {
                            continue;
                        };
                        pair
                    } else {
                        let (q, w) = closest_triangle(
                            p,
                            self.vertices[a],
                            self.vertices[b],
                            self.vertices[c],
                        );
                        ((p - q).norm_squared(), w)
                    };
                    if dist < best || (!float32 && dist == best && f < face) {
                        best = dist;
                        face = f;
                        bary = w;
                    }
                }
            } else {
                let dl = distance(&self.nodes[node.left]);
                let dr = distance(&self.nodes[node.right]);
                if dl < dr {
                    if !float32 || dr < best {
                        stack.push(node.right);
                    }
                    if !float32 || dl < best {
                        stack.push(node.left);
                    }
                } else {
                    if !float32 || dl < best {
                        stack.push(node.left);
                    }
                    if !float32 || dr < best {
                        stack.push(node.right);
                    }
                }
            }
        }
        (best.sqrt(), face, bary)
    }
    pub fn project(&self, points: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        ensure(
            points.shape.len() == 2 && points.shape[1] == 3,
            "projection points need [N,3]",
        )?;
        points.validate()?;
        let mut distances = Vec::new();
        let mut ids = Vec::new();
        let mut weights = Vec::new();
        for p in points.data.chunks_exact(3) {
            let (d, f, w) = self.closest(vec3(p));
            distances.push(d);
            ids.push(f);
            weights.extend(w);
        }
        Ok((
            Tensor::new(vec![ids.len()], distances)?,
            Tensor::indices(vec![ids.len()], ids),
            Tensor::new(vec![points.shape[0], 3], weights)?,
        ))
    }
    pub fn triangle_indices(&self, face: usize) -> [usize; 3] {
        self.faces[face]
    }
    pub fn overlapping_faces(&self, lo: Vec3, hi: Vec3) -> Vec<usize> {
        let mut stack = vec![0];
        let mut result = Vec::new();
        while let Some(i) = stack.pop() {
            let node = &self.nodes[i];
            if (0..3).any(|k| node.hi[k] < lo[k] || node.lo[k] > hi[k]) {
                continue;
            }
            if node.left == usize::MAX {
                result.extend_from_slice(&self.order[node.start..node.end]);
            } else {
                stack.push(node.left);
                stack.push(node.right);
            }
        }
        result
    }
}
