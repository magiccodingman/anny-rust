//! Base glTF morph-channel and metallic-roughness material authoring.
use super::{array_mut, bad, index, mime, GltfAsset};
use crate::{ensure, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MorphDeltas {
    pub positions: Vec<[f64; 3]>,
    pub normals: Vec<[f64; 3]>,
    /// Three-component deltas; the base tangent's handedness is unchanged.
    pub tangents: Vec<[f64; 3]>,
}
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WrapMode {
    #[default]
    Repeat,
    MirroredRepeat,
    ClampToEdge,
}
impl WrapMode {
    fn code(self) -> u32 {
        match self {
            Self::Repeat => 10497,
            Self::MirroredRepeat => 33648,
            Self::ClampToEdge => 33071,
        }
    }
}
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TextureFilter {
    Nearest,
    #[default]
    Linear,
}
impl TextureFilter {
    fn code(self) -> u32 {
        match self {
            Self::Nearest => 9728,
            Self::Linear => 9729,
        }
    }
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TextureOptions {
    pub name: String,
    pub wrap_s: WrapMode,
    pub wrap_t: WrapMode,
    pub filter: TextureFilter,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TextureReference {
    pub texture: usize,
    #[serde(default)]
    pub tex_coord: u32,
}
impl TextureReference {
    fn value(&self) -> Value {
        json!({"index":self.texture,"texCoord":self.tex_coord})
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PbrMaterial {
    pub name: String,
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: [f32; 3],
    pub double_sided: bool,
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: f32,
    pub base_color_texture: Option<TextureReference>,
    pub metallic_roughness_texture: Option<TextureReference>,
    pub normal_texture: Option<TextureReference>,
    pub normal_scale: f32,
    pub occlusion_texture: Option<TextureReference>,
    pub occlusion_strength: f32,
    pub emissive_texture: Option<TextureReference>,
}
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum AlphaMode {
    #[default]
    Opaque,
    Mask,
    Blend,
}
impl Default for PbrMaterial {
    fn default() -> Self {
        Self {
            name: String::new(),
            base_color: [1.; 4],
            metallic: 0.,
            roughness: 1.,
            emissive: [0.; 3],
            double_sided: false,
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            base_color_texture: None,
            metallic_roughness_texture: None,
            normal_texture: None,
            normal_scale: 1.,
            occlusion_texture: None,
            occlusion_strength: 1.,
            emissive_texture: None,
        }
    }
}
impl GltfAsset {
    /// Append one morph channel to every primitive of a mesh. Deltas use the
    /// mesh-local glTF coordinate frame and already-split primitive vertex order.
    /// Adding channels to a mesh with weight animation is rejected: do it before
    /// authoring animation clips, so no animation stride can become stale.
    pub fn add_morph_target(
        &mut self,
        mesh: usize,
        name: &str,
        deltas: &[MorphDeltas],
        weight: f64,
    ) -> Result<usize> {
        ensure(
            !name.is_empty() && weight.is_finite(),
            "morph name/weight invalid",
        )?;
        let mesh_value = self.graph.root["meshes"]
            .as_array()
            .and_then(|a| a.get(mesh))
            .ok_or_else(|| bad("mesh index out of range"))?;
        let primitives = mesh_value["primitives"]
            .as_array()
            .ok_or_else(|| bad("mesh has no primitives"))?;
        ensure(
            !primitives.is_empty() && deltas.len() == primitives.len(),
            "one morph delta set is required per primitive",
        )?;
        let count = primitives[0]
            .get("targets")
            .map(|v| {
                v.as_array()
                    .map(Vec::len)
                    .ok_or_else(|| bad("invalid morph target array"))
            })
            .transpose()?
            .unwrap_or(0);
        ensure(count < 10_000, "too many morph targets")?;
        let mut names: Vec<String> =
            match mesh_value.get("extras").and_then(|e| e.get("targetNames")) {
                Some(n) => serde_json::from_value(n.clone())?,
                None => (0..count).map(|i| format!("target-{i}")).collect(),
            };
        ensure(
            names.len() == count && !names.iter().any(|n| n == name),
            "morph name count mismatch or duplicate name",
        )?;
        names.push(name.to_owned());
        for (p, d) in primitives.iter().zip(deltas) {
            let n = self
                .graph
                .accessor(index(&p["attributes"], "POSITION")?, 3)?
                .len()
                / 3;
            ensure(
                p.get("targets")
                    .map_or(Some(0), |t| t.as_array().map(Vec::len))
                    == Some(count),
                "mesh primitives have different morph counts",
            )?;
            ensure(
                d.positions.len() == n && d.positions.iter().flatten().all(|x| x.is_finite()),
                "morph position count/values invalid",
            )?;
            for (v, attr, width) in [(&d.normals, "NORMAL", 3), (&d.tangents, "TANGENT", 4)] {
                if !v.is_empty() {
                    ensure(
                        v.len() == n && v.iter().flatten().all(|x| x.is_finite()),
                        "morph attribute count/values invalid",
                    )?;
                    ensure(
                        self.graph
                            .accessor(index(&p["attributes"], attr)?, width)?
                            .len()
                            == n * width,
                        "missing matching base morph attribute",
                    )?;
                }
            }
        }
        for clip in self.animation_clips()? {
            for c in clip.channels {
                if c.path == super::AnimationPath::Weights
                    && self.graph.root["nodes"][c.node]["mesh"].as_u64() == Some(mesh as u64)
                {
                    return Err(bad(
                        "append morph targets before authoring weight animation",
                    ));
                }
            }
        }
        let mut weights: Vec<f64> = mesh_value
            .get("weights")
            .map(|v| serde_json::from_value(v.clone()))
            .transpose()?
            .unwrap_or_else(|| vec![0.; count]);
        ensure(
            weights.len() == count && weights.iter().all(|x| x.is_finite()),
            "invalid default morph weights",
        )?;
        weights.push(weight);
        let mut next = self.clone();
        for (i, d) in deltas.iter().enumerate() {
            let mut target = json!({});
            for (v, attr, bounds) in [
                (&d.positions, "POSITION", true),
                (&d.normals, "NORMAL", false),
                (&d.tangents, "TANGENT", false),
            ] {
                if !v.is_empty() {
                    let a = next.append_floats(
                        &v.iter().flatten().copied().collect::<Vec<_>>(),
                        3,
                        "VEC3",
                        bounds,
                    )?;
                    target[attr] = json!(a);
                }
            }
            array_mut(
                &mut next.graph.root["meshes"][mesh]["primitives"][i],
                "targets",
            )?
            .push(target);
        }
        next.graph.root["meshes"][mesh]["weights"] = json!(weights);
        let extras = &mut next.graph.root["meshes"][mesh]["extras"];
        if extras.is_null() {
            *extras = json!({});
        }
        ensure(
            extras.is_object(),
            "mesh extras must be an object to add target names",
        )?;
        extras["targetNames"] = json!(names);
        for node in next.graph.root["nodes"]
            .as_array_mut()
            .ok_or_else(|| bad("missing nodes"))?
        {
            if node["mesh"].as_u64() == Some(mesh as u64) {
                if let Some(w) = node.get_mut("weights") {
                    let a = w
                        .as_array_mut()
                        .ok_or_else(|| bad("invalid node weights"))?;
                    ensure(a.len() == count, "node morph weight count mismatch")?;
                    a.push(json!(weight));
                }
            }
        }
        next.geometry()?;
        *self = next;
        Ok(count)
    }
    pub fn add_texture(&mut self, bytes: &[u8], options: &TextureOptions) -> Result<usize> {
        let kind = mime(bytes)?;
        let mut next = self.clone();
        let view = next.append_bytes(bytes.to_vec())?;
        let images = array_mut(&mut next.graph.root, "images")?;
        let image = images.len();
        images.push(json!({"name":options.name,"bufferView":view,"mimeType":kind}));
        let samplers = array_mut(&mut next.graph.root, "samplers")?;
        let sampler = samplers.len();
        samplers.push(json!({"wrapS":options.wrap_s.code(),"wrapT":options.wrap_t.code(),"minFilter":options.filter.code(),"magFilter":options.filter.code()}));
        let textures = array_mut(&mut next.graph.root, "textures")?;
        let texture = textures.len();
        textures.push(json!({"name":options.name,"source":image,"sampler":sampler}));
        *self = next;
        Ok(texture)
    }
    pub fn set_material(
        &mut self,
        mesh: usize,
        primitive: usize,
        material: &PbrMaterial,
    ) -> Result<usize> {
        let p = self.graph.root["meshes"]
            .as_array()
            .and_then(|v| v.get(mesh))
            .and_then(|m| m["primitives"].as_array())
            .and_then(|p| p.get(primitive))
            .ok_or_else(|| bad("material mesh/primitive out of range"))?;
        ensure(
            material
                .base_color
                .iter()
                .chain(&material.emissive)
                .all(|x| x.is_finite() && (0.0..=1.0).contains(x))
                && [
                    material.metallic,
                    material.roughness,
                    material.occlusion_strength,
                ]
                .iter()
                .all(|x| x.is_finite() && (0.0..=1.0).contains(x))
                && material.normal_scale.is_finite()
                && material.alpha_cutoff.is_finite()
                && material.alpha_cutoff >= 0.,
            "invalid PBR material factors",
        )?;
        for t in [
            &material.base_color_texture,
            &material.metallic_roughness_texture,
            &material.normal_texture,
            &material.occlusion_texture,
            &material.emissive_texture,
        ]
        .into_iter()
        .flatten()
        {
            ensure(
                self.graph.root["textures"]
                    .as_array()
                    .is_some_and(|v| t.texture < v.len()),
                "material texture index out of range",
            )?;
            let attribute = format!("TEXCOORD_{}", t.tex_coord);
            ensure(
                p["attributes"].get(&attribute).is_some(),
                "textured material requires its UV set",
            )?;
        }
        let mut value = json!({"name":material.name,"doubleSided":material.double_sided,"alphaMode":material.alpha_mode,
            "emissiveFactor":material.emissive,"pbrMetallicRoughness":{"baseColorFactor":material.base_color,"metallicFactor":material.metallic,"roughnessFactor":material.roughness}});
        if matches!(material.alpha_mode, AlphaMode::Mask) {
            value["alphaCutoff"] = json!(material.alpha_cutoff);
        }
        for (reference, key) in [
            (&material.base_color_texture, "baseColorTexture"),
            (
                &material.metallic_roughness_texture,
                "metallicRoughnessTexture",
            ),
        ] {
            if let Some(t) = reference {
                value["pbrMetallicRoughness"][key] = t.value();
            }
        }
        if let Some(t) = &material.normal_texture {
            let mut v = t.value();
            v["scale"] = json!(material.normal_scale);
            value["normalTexture"] = v;
        }
        if let Some(t) = &material.occlusion_texture {
            let mut v = t.value();
            v["strength"] = json!(material.occlusion_strength);
            value["occlusionTexture"] = v;
        }
        if let Some(t) = &material.emissive_texture {
            value["emissiveTexture"] = t.value();
        }
        let materials = array_mut(&mut self.graph.root, "materials")?;
        let id = materials.len();
        materials.push(value);
        self.graph.root["meshes"][mesh]["primitives"][primitive]["material"] = json!(id);
        Ok(id)
    }
}
