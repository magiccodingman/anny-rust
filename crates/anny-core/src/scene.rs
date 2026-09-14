//! Portable scene construction and glTF 2.0/GLB export, without Python.
//!
//! The numerical model remains Z-up in meters. A single scene root rotates it
//! into glTF's Y-up space; skin bind matrices and vertex data stay consistent.
//! Static export bakes the selected pose. Rigged export uses the shaped rest mesh
//! and ordinary glTF linear skinning (DQS must be baked instead).
use crate::{ensure, math::*, Anny, Error, Parameters, Result, SkinningMethod, Tensor};
use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SurfaceMesh {
    pub positions: Vec<[f64; 3]>,
    pub triangles: Vec<[u32; 3]>,
    #[serde(default)]
    pub normals: Vec<[f64; 3]>,
    #[serde(default)]
    pub texcoords: Vec<[f64; 2]>,
    /// Optional mapping back to the unsplit model vertices (UV seams duplicate vertices).
    #[serde(default)]
    pub source_vertex_indices: Vec<u32>,
}
impl SurfaceMesh {
    pub fn validate(&self) -> Result<()> {
        ensure(!self.positions.is_empty(), "mesh has no vertices")?;
        ensure(
            self.positions.iter().flatten().all(|x| x.is_finite()),
            "non-finite mesh position",
        )?;
        ensure(
            self.triangles
                .iter()
                .flatten()
                .all(|&i| (i as usize) < self.positions.len()),
            "mesh index out of range",
        )?;
        for (count, what) in [
            (self.normals.len(), "normal"),
            (self.texcoords.len(), "UV"),
            (self.source_vertex_indices.len(), "source vertex"),
        ] {
            ensure(
                count == 0 || count == self.positions.len(),
                format!("{what} count mismatch"),
            )?;
        }
        ensure(
            self.normals
                .iter()
                .flatten()
                .chain(self.texcoords.iter().flatten())
                .all(|x| x.is_finite()),
            "non-finite mesh attribute",
        )?;
        Ok(())
    }
    pub fn vertex_tensor(&self) -> Result<Tensor> {
        self.validate()?;
        Tensor::new(
            vec![self.positions.len(), 3],
            self.positions.iter().flatten().copied().collect(),
        )
    }
    pub fn face_tensor(&self) -> Tensor {
        Tensor::indices(
            vec![self.triangles.len(), 3],
            self.triangles
                .iter()
                .flatten()
                .map(|&x| x as usize)
                .collect(),
        )
    }
    /// Area-weighted smooth normals; unused/degenerate vertices get a unit fallback.
    pub fn recalculate_normals(&mut self) -> Result<()> {
        self.validate()?;
        let mut normals = vec![Vec3::zeros(); self.positions.len()];
        for f in &self.triangles {
            let [a, b, c] = f.map(|i| vec3(&self.positions[i as usize]));
            let n = (b - a).cross(&(c - a));
            for &i in f {
                normals[i as usize] += n;
            }
        }
        self.normals = normals
            .iter()
            .map(|v| {
                let n = v.try_normalize(1e-30).unwrap_or_else(Vec3::z);
                [n.x, n.y, n.z]
            })
            .collect();
        Ok(())
    }
}
#[derive(Clone, Debug)]
pub struct Skin {
    pub names: Vec<String>,
    pub parents: Vec<i32>,
    pub rest: Vec<Mat4>,
    pub pose: Vec<Mat4>,
    pub indices: Vec<Vec<u16>>,
    pub weights: Vec<Vec<f64>>,
}
#[derive(Clone, Debug)]
pub struct SceneObject {
    pub name: String,
    pub mesh: SurfaceMesh,
    /// Row/column semantics match the core: multiply column vectors.
    pub transform: Mat4,
    pub color: [f32; 4],
    pub skin: Option<Skin>,
    pub extras: Value,
}
#[derive(Clone, Debug)]
pub struct Animation {
    pub name: String,
    pub object: usize,
    pub times: Vec<f64>,
    /// Absolute bone poses in the same Z-up frame as the object's bind skeleton.
    pub bone_poses: Vec<Vec<Mat4>>,
}
#[derive(Clone, Debug, Default)]
pub struct Scene {
    pub objects: Vec<SceneObject>,
    pub animations: Vec<Animation>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CharacterExport {
    pub name: String,
    pub translation: [f64; 3],
    pub color: [f32; 4],
    pub rigged: bool,
    pub batch_index: usize,
}
impl Default for CharacterExport {
    fn default() -> Self {
        Self {
            name: "Anny".into(),
            translation: [0.; 3],
            color: [0.7, 0.7, 0.7, 1.],
            rigged: false,
            batch_index: 0,
        }
    }
}
fn selected(t: &Tensor, b: usize, trailing: &[usize], name: &str) -> Result<Vec<f64>> {
    ensure(
        t.shape.len() == trailing.len() + 1 && t.shape[1..] == *trailing,
        format!("invalid {name} shape"),
    )?;
    let batch = if t.shape[0] == 1 { 0 } else { b };
    ensure(
        batch < t.shape[0],
        format!("{name} batch index out of range"),
    )?;
    let stride: usize = trailing.iter().product();
    Ok(t.data[batch * stride..(batch + 1) * stride].to_vec())
}
fn smooth_mesh(model: &Anny, vertices: &[f64]) -> Result<SurfaceMesh> {
    // Select diagonals from the template, as Anny does, never from the posed mesh.
    let mut topology = crate::ModelData::default();
    for key in [
        "template_vertices",
        "faces",
        "texture_coordinates",
        "face_texture_coordinate_indices",
    ] {
        if let Some(t) = model.data.arrays.get(key) {
            topology.put(key, t.clone());
        }
    }
    crate::mesh::triangulate(&mut topology)?;
    let faces = topology.get("faces")?;
    let mut base = SurfaceMesh {
        positions: vertices
            .chunks_exact(3)
            .map(|v| [v[0], v[1], v[2]])
            .collect(),
        triangles: faces
            .checked_indices(model.data.vertex_count(), "faces")?
            .chunks_exact(3)
            .map(|f| [f[0] as u32, f[1] as u32, f[2] as u32])
            .collect(),
        ..Default::default()
    };
    base.recalculate_normals()?;
    let Some(uv) = topology.arrays.get("texture_coordinates") else {
        base.source_vertex_indices = (0..base.positions.len()).map(|i| i as u32).collect();
        return Ok(base);
    };
    let uv_faces = topology.get("face_texture_coordinate_indices")?;
    let ui = uv_faces.checked_indices(uv.shape[0], "face UV indices")?;
    let mut pairs = BTreeMap::new();
    let mut out = SurfaceMesh::default();
    for (f, face) in base.triangles.iter().enumerate() {
        let mut tri = [0; 3];
        for (s, &v) in face.iter().enumerate() {
            let u = ui[f * 3 + s];
            let index = if let Some(&index) = pairs.get(&(v, u)) {
                index
            } else {
                let index = u32::try_from(out.positions.len())
                    .map_err(|_| Error::Invalid("too many scene vertices".into()))?;
                pairs.insert((v, u), index);
                out.positions.push(base.positions[v as usize]);
                out.normals.push(base.normals[v as usize]);
                // glTF V=0 is the top of the image; OBJ/Anny V=0 is the bottom.
                out.texcoords
                    .push([uv.data[u * 2], 1. - uv.data[u * 2 + 1]]);
                out.source_vertex_indices.push(v);
                index
            };
            tri[s] = index;
        }
        out.triangles.push(tri);
    }
    out.validate()?;
    Ok(out)
}
impl Scene {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add_character(
        &mut self,
        model: &Anny,
        parameters: &Parameters,
        options: &CharacterExport,
    ) -> Result<usize> {
        let output = model.forward(parameters)?;
        self.add_evaluated_character(model, parameters, options, &output)
    }
    /// Pack an already evaluated character (including a widened f32 result).
    /// The output must belong to this model/configuration and parameter set.
    /// This method performs no forward evaluation; it checks array dimensions.
    pub fn add_evaluated_character(
        &mut self,
        model: &Anny,
        parameters: &Parameters,
        options: &CharacterExport,
        output: &crate::ModelOutput,
    ) -> Result<usize> {
        ensure(
            !options.rigged || model.config.skinning_method != SkinningMethod::Dqs,
            "glTF skins use LBS, not DQS; export a baked mesh or select LBS",
        )?;
        ensure(
            options.translation.iter().all(|x| x.is_finite()),
            "invalid scene translation",
        )?;
        for tensor in output.arrays.values() {
            tensor.validate()?;
        }
        let n = model.data.vertex_count();
        ensure(
            options.batch_index < output.get("vertices")?.shape[0],
            "character batch index out of range",
        )?;
        let key = if options.rigged {
            "rest_vertices"
        } else {
            "vertices"
        };
        let vertices = selected(output.get(key)?, options.batch_index, &[n, 3], key)?;
        let mesh = smooth_mesh(model, &vertices)?;
        let skin = if options.rigged {
            let j = model.data.bone_count();
            ensure(j <= u16::MAX as usize + 1, "glTF joint count exceeds u16")?;
            let rest = selected(
                output.get("rest_bone_poses")?,
                options.batch_index,
                &[j, 4, 4],
                "rest bones",
            )?;
            let pose = selected(
                output.get("bone_poses")?,
                options.batch_index,
                &[j, 4, 4],
                "bones",
            )?;
            let weights = model.data.get("vertex_bone_weights")?;
            let indices = model.data.get("vertex_bone_indices")?;
            let width = weights.shape[1];
            let mut wi = Vec::new();
            let mut ww = Vec::new();
            for &source in &mesh.source_vertex_indices {
                let off = source as usize * width;
                wi.push(
                    indices.data[off..off + width]
                        .iter()
                        .map(|&x| x as u16)
                        .collect(),
                );
                ww.push(weights.data[off..off + width].to_vec());
            }
            Some(Skin {
                names: model.data.metadata.bone_labels.clone(),
                parents: model.data.metadata.bone_parents.clone(),
                rest: rest.chunks_exact(16).map(mat4).collect(),
                pose: pose.chunks_exact(16).map(mat4).collect(),
                indices: wi,
                weights: ww,
            })
        } else {
            None
        };
        let i = self.objects.len();
        self.objects.push(SceneObject {
            name:options.name.clone(),mesh,transform:rigid(&Mat3::identity(), &vec3(&options.translation)), color:options.color,skin,
            extras:json!({"anny":{"upstream":crate::UPSTREAM_REVISION,"config":model.config,"parameters":parameters,"batchIndex":options.batch_index,"baked":!options.rigged}}),
        });
        Ok(i)
    }
    /// Attach sampled poses to an already added rigged character. Shape stays fixed.
    pub fn add_animation(&mut self, clip: Animation) -> Result<()> {
        let object = self
            .objects
            .get(clip.object)
            .ok_or_else(|| Error::Invalid("animation object out of range".into()))?;
        let skin = object
            .skin
            .as_ref()
            .ok_or_else(|| Error::Invalid("animation needs a rigged object".into()))?;
        ensure(
            !clip.times.is_empty() && clip.times.len() == clip.bone_poses.len(),
            "animation frame count mismatch",
        )?;
        ensure(
            clip.times.iter().all(|&t| t.is_finite() && t >= 0.)
                && clip.times.windows(2).all(|w| (w[1] as f32) > (w[0] as f32)),
            "animation times must be increasing finite f32 seconds",
        )?;
        for poses in &clip.bone_poses {
            ensure(
                poses.len() == skin.names.len(),
                "animation joint count mismatch",
            )?;
            for p in poses {
                check_scene_rigid(p)?;
            }
        }
        self.animations.push(clip);
        Ok(())
    }
    pub fn to_glb(&self) -> Result<Vec<u8>> {
        let (root, mut bin) = self.document()?;
        let mut js = serde_json::to_vec(&root)?;
        while js.len() % 4 != 0 {
            js.push(b' ');
        }
        while bin.len() % 4 != 0 {
            bin.push(0);
        }
        let len = 12usize
            .checked_add(8)
            .and_then(|x| x.checked_add(js.len()))
            .and_then(|x| x.checked_add(8))
            .and_then(|x| x.checked_add(bin.len()))
            .ok_or_else(|| Error::Invalid("GLB length overflow".into()))?;
        let len = u32::try_from(len).map_err(|_| Error::Invalid("GLB exceeds 4 GiB".into()))?;
        let mut out = Vec::with_capacity(len as usize);
        out.extend(b"glTF");
        out.extend(2u32.to_le_bytes());
        out.extend(len.to_le_bytes());
        out.extend((js.len() as u32).to_le_bytes());
        out.extend(b"JSON");
        out.extend(js);
        out.extend((bin.len() as u32).to_le_bytes());
        out.extend(b"BIN\0");
        out.extend(bin);
        Ok(out)
    }
    /// Embedded-buffer glTF JSON, so a file/byte result has no external .bin dependency.
    pub fn to_gltf(&self) -> Result<Vec<u8>> {
        let (mut doc, bin) = self.document()?;
        doc["buffers"][0]["uri"] = json!(format!(
            "data:application/octet-stream;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(bin)
        ));
        Ok(serde_json::to_vec_pretty(&doc)?)
    }
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let bytes = match ext.as_str() {
            "glb" => self.to_glb()?,
            "gltf" => self.to_gltf()?,
            _ => return Err(Error::Invalid("scene export requires .glb or .gltf".into())),
        };
        std::fs::write(path, bytes)?;
        Ok(())
    }
    fn document(&self) -> Result<(Value, Vec<u8>)> {
        ensure(!self.objects.is_empty(), "scene has no objects")?;
        let mut b = GltfBuilder::default();
        let q = quaternion(&rotvec(&Vec3::new(-std::f64::consts::FRAC_PI_2, 0., 0.)));
        b.nodes
            .push(json!({"name":"Anny Z-up to glTF Y-up","rotation":q,"children":[]}));
        let mut roots = Vec::new();
        let mut skinned_roots = Vec::new();
        let mut joint_nodes = Vec::new();
        for obj in &self.objects {
            obj.mesh.validate()?;
            ensure(
                !obj.mesh.triangles.is_empty(),
                "glTF triangle mesh is empty",
            )?;
            ensure(
                obj.color
                    .iter()
                    .all(|x| x.is_finite() && (0.0..=1.0).contains(x)),
                "material color outside [0,1]",
            )?;
            check_scene_rigid(&obj.transform)?;
            let container = b.nodes.len();
            roots.push(container);
            b.nodes.push(json!({"name":obj.name,"children":[]}));
            if obj.transform != Mat4::identity() {
                b.nodes[container]["matrix"] = json!(obj.transform.as_slice());
            }
            let mesh_node = b.nodes.len();
            b.nodes
                .push(json!({"name":format!("{} mesh",obj.name),"mesh":b.meshes.len()}));
            let mut children = if obj.skin.is_some() {
                skinned_roots.push(mesh_node);
                Vec::new()
            } else {
                vec![mesh_node]
            };
            let mut attributes = serde_json::Map::new();
            let positions = obj
                .mesh
                .positions
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>();
            attributes.insert(
                "POSITION".into(),
                json!(b.floats(&positions, 3, "VEC3", Some(34962), true)?),
            );
            let normals = if obj.mesh.normals.is_empty() {
                let mut m = obj.mesh.clone();
                m.recalculate_normals()?;
                m.normals
            } else {
                obj.mesh.normals.clone()
            };
            let normals = normals
                .iter()
                .flat_map(|n| {
                    let n = vec3(n).try_normalize(1e-30).unwrap_or_else(Vec3::z);
                    [n.x, n.y, n.z]
                })
                .collect::<Vec<_>>();
            attributes.insert(
                "NORMAL".into(),
                json!(b.floats(&normals, 3, "VEC3", Some(34962), false)?),
            );
            if !obj.mesh.texcoords.is_empty() {
                attributes.insert(
                    "TEXCOORD_0".into(),
                    json!(b.floats(
                        &obj.mesh
                            .texcoords
                            .iter()
                            .flatten()
                            .copied()
                            .collect::<Vec<_>>(),
                        2,
                        "VEC2",
                        Some(34962),
                        false
                    )?),
                );
            }
            let mut nodes = Vec::new();
            if let Some(skin) = &obj.skin {
                let j = skin.names.len();
                ensure(
                    j > 0
                        && skin.parents.len() == j
                        && skin.rest.len() == j
                        && skin.pose.len() == j,
                    "skin metadata count mismatch",
                )?;
                propagation_order(&skin.parents)?;
                ensure(
                    skin.weights.len() == obj.mesh.positions.len()
                        && skin.indices.len() == obj.mesh.positions.len(),
                    "skin vertex count mismatch",
                )?;
                let start = b.nodes.len();
                for k in 0..j {
                    check_scene_rigid(&skin.rest[k])?;
                    check_scene_rigid(&skin.pose[k])?;
                    let p = skin.parents[k];
                    let local = if p < 0 {
                        skin.pose[k]
                    } else {
                        inverse_rigid(&skin.pose[p as usize]) * skin.pose[k]
                    };
                    b.nodes.push(pose_node(&skin.names[k], &local)?);
                    nodes.push(start + k);
                }
                for k in 0..j {
                    if skin.parents[k] < 0 {
                        children.push(start + k);
                    } else {
                        let p = start + skin.parents[k] as usize;
                        if b.nodes[p].get("children").is_none() {
                            b.nodes[p]["children"] = json!([]);
                        }
                        b.nodes[p]["children"]
                            .as_array_mut()
                            .unwrap()
                            .push(json!(start + k));
                    }
                }
                let inv = skin
                    .rest
                    .iter()
                    .flat_map(|p| inverse_rigid(p).as_slice().to_vec())
                    .collect::<Vec<_>>();
                let ibm = b.floats(&inv, 16, "MAT4", None, false)?;
                let skin_id = b.skins.len();
                b.skins.push(json!({"name":format!("{} skeleton",obj.name),"joints":nodes,"inverseBindMatrices":ibm}));
                b.nodes[mesh_node]["skin"] = json!(skin_id);
                let width = skin.weights.iter().map(Vec::len).max().unwrap_or(0);
                ensure(width > 0, "skin has no influences")?;
                for (w, i) in skin.weights.iter().zip(&skin.indices) {
                    ensure(
                        w.len() == i.len()
                            && w.iter().all(|x| x.is_finite() && *x >= 0.)
                            && (w.iter().sum::<f64>() - 1.).abs() < 1e-5
                            && i.iter().all(|&x| (x as usize) < j),
                        "invalid glTF skin influences",
                    )?;
                }
                for set in 0..width.div_ceil(4) {
                    let mut indices = Vec::new();
                    let mut weights = Vec::new();
                    for (w, i) in skin.weights.iter().zip(&skin.indices) {
                        for slot in set * 4..set * 4 + 4 {
                            indices.push(if *w.get(slot).unwrap_or(&0.) == 0. {
                                0
                            } else {
                                *i.get(slot).unwrap_or(&0)
                            });
                            weights.push(*w.get(slot).unwrap_or(&0.));
                        }
                    }
                    attributes.insert(
                        format!("JOINTS_{set}"),
                        json!(b.u16s(&indices, 4, "VEC4", Some(34962))),
                    );
                    attributes.insert(
                        format!("WEIGHTS_{set}"),
                        json!(b.floats(&weights, 4, "VEC4", Some(34962), false)?),
                    );
                }
            }
            joint_nodes.push(nodes);
            b.nodes[container]["children"] = json!(children);
            let indices = b.u32s(
                &obj.mesh
                    .triangles
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<Vec<_>>(),
                Some(34963),
            );
            let mat = b.materials.len();
            b.materials.push(json!({"name":format!("{} material",obj.name),"pbrMetallicRoughness":{"baseColorFactor":obj.color,"metallicFactor":0.0,"roughnessFactor":0.8},"doubleSided":true,"alphaMode":if obj.color[3]<1. {"BLEND"}else{"OPAQUE"}}));
            let mut extras = obj.extras.clone();
            if !extras.is_object() {
                extras = json!({});
            }
            if !obj.mesh.source_vertex_indices.is_empty() {
                extras["sourceVertexIndices"] = json!(obj.mesh.source_vertex_indices);
            }
            b.meshes.push(json!({"name":obj.name,"primitives":[{"attributes":attributes,"indices":indices,"material":mat,"mode":4}],"extras":extras}));
        }
        b.nodes[0]["children"] = json!(roots);
        for clip in &self.animations {
            ensure(clip.object < self.objects.len(), "invalid animation object")?;
            let skin = self.objects[clip.object]
                .skin
                .as_ref()
                .ok_or_else(|| Error::Invalid("animation without skin".into()))?;
            // Revalidate here because scene/clip fields are public.
            ensure(
                !clip.times.is_empty()
                    && clip.times.len() == clip.bone_poses.len()
                    && clip.bone_poses.iter().all(|p| p.len() == skin.names.len()),
                "animation frame count mismatch",
            )?;
            ensure(
                clip.times.iter().all(|&t| t >= 0. && t.is_finite())
                    && clip.times.windows(2).all(|w| (w[1] as f32) > (w[0] as f32)),
                "invalid animation times",
            )?;
            let input = b.floats(&clip.times, 1, "SCALAR", None, true)?;
            let mut samplers = Vec::new();
            let mut channels = Vec::new();
            for bone in 0..skin.names.len() {
                let mut tr = Vec::new();
                let mut ro = Vec::new();
                let mut prev = [0., 0., 0., 1.];
                for frame in &clip.bone_poses {
                    check_scene_rigid(&frame[bone])?;
                    let p = skin.parents[bone];
                    let local = if p < 0 {
                        frame[bone]
                    } else {
                        inverse_rigid(&frame[p as usize]) * frame[bone]
                    };
                    tr.extend(translation(&local).iter().copied());
                    let mut q = quaternion(&rotation(&local));
                    if q.iter().zip(prev).map(|(a, b)| a * b).sum::<f64>() < 0. {
                        for x in &mut q {
                            *x = -*x;
                        }
                    }
                    ro.extend(q);
                    prev = q;
                }
                for (values, width, ty, path) in [
                    (&tr, 3, "VEC3", "translation"),
                    (&ro, 4, "VEC4", "rotation"),
                ] {
                    let output = b.floats(values, width, ty, None, false)?;
                    channels.push(json!({"sampler":samplers.len(),"target":{"node":joint_nodes[clip.object][bone],"path":path}}));
                    samplers.push(json!({"input":input,"output":output,"interpolation":"LINEAR"}));
                }
            }
            b.animations
                .push(json!({"name":clip.name,"samplers":samplers,"channels":channels}));
        }
        let mut scene_roots = vec![0usize];
        scene_roots.extend(skinned_roots);
        let mut root = json!({"asset":{"version":"2.0","generator":"anny-rust"},"scene":0,"scenes":[{"nodes":scene_roots}],"nodes":b.nodes,"meshes":b.meshes,"materials":b.materials,"buffers":[{"byteLength":b.bin.len()}],"bufferViews":b.views,"accessors":b.accessors});
        if !b.skins.is_empty() {
            root["skins"] = json!(b.skins);
        }
        if !b.animations.is_empty() {
            root["animations"] = json!(b.animations);
        }
        Ok((root, b.bin))
    }
}
fn pose_node(name: &str, m: &Mat4) -> Result<Value> {
    check_scene_rigid(m)?;
    Ok(
        json!({"name":name,"translation":translation(m).as_slice(),"rotation":quaternion(&rotation(m))}),
    )
}
#[derive(Default)]
struct GltfBuilder {
    bin: Vec<u8>,
    views: Vec<Value>,
    accessors: Vec<Value>,
    nodes: Vec<Value>,
    meshes: Vec<Value>,
    skins: Vec<Value>,
    materials: Vec<Value>,
    animations: Vec<Value>,
}
impl GltfBuilder {
    fn view(&mut self, bytes: &[u8], target: Option<u32>) -> usize {
        while !self.bin.len().is_multiple_of(4) {
            self.bin.push(0);
        }
        let id = self.views.len();
        let mut v = json!({"buffer":0,"byteOffset":self.bin.len(),"byteLength":bytes.len()});
        if let Some(t) = target {
            v["target"] = json!(t);
        }
        self.bin.extend(bytes);
        self.views.push(v);
        id
    }
    fn floats(
        &mut self,
        values: &[f64],
        width: usize,
        ty: &str,
        target: Option<u32>,
        bounds: bool,
    ) -> Result<usize> {
        ensure(
            !values.is_empty() && values.len().is_multiple_of(width),
            "invalid glTF accessor length",
        )?;
        let values = values.iter().map(|&x| x as f32).collect::<Vec<_>>();
        ensure(
            values.iter().all(|x| x.is_finite()),
            "glTF value outside finite f32 range",
        )?;
        let bytes = values
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>();
        let view = self.view(&bytes, target);
        let mut a =
            json!({"bufferView":view,"componentType":5126,"count":values.len()/width,"type":ty});
        if bounds {
            let mut min = vec![f32::INFINITY; width];
            let mut max = vec![f32::NEG_INFINITY; width];
            for row in values.chunks_exact(width) {
                for i in 0..width {
                    min[i] = min[i].min(row[i]);
                    max[i] = max[i].max(row[i]);
                }
            }
            a["min"] = json!(min);
            a["max"] = json!(max);
        }
        let id = self.accessors.len();
        self.accessors.push(a);
        Ok(id)
    }
    fn u16s(&mut self, values: &[u16], width: usize, ty: &str, target: Option<u32>) -> usize {
        let bytes = values
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>();
        let view = self.view(&bytes, target);
        let id = self.accessors.len();
        self.accessors.push(
            json!({"bufferView":view,"componentType":5123,"count":values.len()/width,"type":ty}),
        );
        id
    }
    fn u32s(&mut self, values: &[u32], target: Option<u32>) -> usize {
        let bytes = values
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>();
        let view = self.view(&bytes, target);
        let id = self.accessors.len();
        self.accessors.push(
            json!({"bufferView":view,"componentType":5125,"count":values.len(),"type":"SCALAR"}),
        );
        id
    }
}

fn check_scene_rigid(m: &Mat4) -> Result<()> {
    crate::math::checked_rigid(m)?;
    let r = rotation(m);
    ensure(
        (r.transpose() * r - Mat3::identity()).amax() < 1e-5 && (r.determinant() - 1.).abs() < 1e-5,
        "scene skeleton/transform must be rigid; bake scale or shear into geometry first",
    )
}
