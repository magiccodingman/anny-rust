//! Shared authoring requests for Rust, native C/C#, CLI and WASM hosts.
use super::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "kebab-case", deny_unknown_fields)]
pub enum GltfEdit {
    AddTexture {
        bytes: Vec<u8>,
        #[serde(default)]
        options: TextureOptions,
    },
    SetMaterial {
        mesh: usize,
        primitive: usize,
        material: Box<PbrMaterial>,
    },
    AddMorph {
        mesh: usize,
        name: String,
        deltas: Vec<MorphDeltas>,
        #[serde(default)]
        weight: f64,
    },
    AddAnimation {
        clip: AnimationClip,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "kebab-case", deny_unknown_fields)]
pub enum GltfQuery {
    Describe,
    Document,
    Clips,
    Geometry {
        #[serde(default)]
        animation: Option<usize>,
        #[serde(default)]
        time: f64,
    },
}
impl GltfAsset {
    /// Transactional edit batch. Returned indices correspond to the appended
    /// texture/material/morph/animation, in request order. Input stays intact on error.
    pub fn apply_edits(&mut self, operations: &[GltfEdit]) -> Result<Vec<usize>> {
        ensure(
            operations.len() <= 10_000,
            "too many glTF authoring operations",
        )?;
        let mut next = self.clone();
        let mut ids = Vec::new();
        for op in operations {
            ids.push(match op {
                GltfEdit::AddTexture { bytes, options } => next.add_texture(bytes, options)?,
                GltfEdit::SetMaterial {
                    mesh,
                    primitive,
                    material,
                } => next.set_material(*mesh, *primitive, material)?,
                GltfEdit::AddMorph {
                    mesh,
                    name,
                    deltas,
                    weight,
                } => next.add_morph_target(*mesh, name, deltas, *weight)?,
                GltfEdit::AddAnimation { clip } => next.add_animation(clip)?,
            });
        }
        *self = next;
        Ok(ids)
    }
    pub fn query(&self, request: &GltfQuery) -> Result<Value> {
        Ok(match request {
            GltfQuery::Describe => {
                let count = |key: &str| self.graph.root[key].as_array().map_or(0, Vec::len);
                json!({"meshes":count("meshes"),"nodes":count("nodes"),"skins":count("skins"),"materials":count("materials"),"textures":count("textures"),"images":count("images"),"animations":count("animations"),"buffer_bytes":self.graph.buffers.iter().map(Vec::len).sum::<usize>()})
            }
            GltfQuery::Document => self.document().clone(),
            GltfQuery::Clips => serde_json::to_value(self.animation_clips()?)?,
            GltfQuery::Geometry { animation, time } => serde_json::to_value(match animation {
                Some(i) => self.geometry_at(*i, *time)?,
                None => self.geometry()?,
            })?,
        })
    }
}
/// Read a self-contained glTF/GLB, apply an edit array and return owned GLB bytes.
pub fn edit_glb(bytes: &[u8], operations_json: &str) -> Result<Vec<u8>> {
    let operations: Vec<GltfEdit> = serde_json::from_str(operations_json)?;
    let mut asset = GltfAsset::from_bytes(bytes)?;
    asset.apply_edits(&operations)?;
    asset.to_glb()
}
pub fn query_glb(bytes: &[u8], request_json: &str) -> Result<String> {
    let request = serde_json::from_str(request_json)?;
    serde_json::to_string(&GltfAsset::from_bytes(bytes)?.query(&request)?).map_err(Into::into)
}
