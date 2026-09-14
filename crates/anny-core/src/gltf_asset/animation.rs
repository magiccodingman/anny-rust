//! Base glTF animation channels, following glTF 2.0 section 3.11 and appendix C.
//! Raw channels use glTF XYZW/Y-up values; sampled geometry is returned Z-up.
use super::{array_mut, bad, index, GltfAsset};
use crate::{ensure, scene::SurfaceMesh, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AnimationPath {
    Translation,
    Rotation,
    Scale,
    Weights,
}
impl AnimationPath {
    fn key(self) -> &'static str {
        match self {
            Self::Translation => "translation",
            Self::Rotation => "rotation",
            Self::Scale => "scale",
            Self::Weights => "weights",
        }
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Interpolation {
    #[default]
    Linear,
    Step,
    CubicSpline,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnimationChannel {
    pub node: usize,
    pub path: AnimationPath,
    #[serde(default)]
    pub interpolation: Interpolation,
    pub times: Vec<f64>,
    /// Component count per value (3/4 for TRS, morph count for weights).
    pub width: usize,
    /// Key-major values. Cubic keys contain [incoming tangent, value, outgoing tangent].
    pub values: Vec<f64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnimationClip {
    pub name: String,
    pub channels: Vec<AnimationChannel>,
}
impl AnimationChannel {
    pub fn validate(&self) -> Result<()> {
        ensure(
            !self.times.is_empty() && self.times.len() <= 1_000_000,
            "invalid animation key count",
        )?;
        ensure(
            self.times.iter().all(|x| x.is_finite() && *x >= 0.)
                && self.times.windows(2).all(|w| w[1] > w[0]),
            "animation times must be finite, nonnegative and strictly increasing",
        )?;
        let factor = if self.interpolation == Interpolation::CubicSpline {
            3usize
        } else {
            1
        };
        ensure(
            factor == 1 || self.times.len() >= 2,
            "cubic animation needs two keys",
        )?;
        let width_ok = match self.path {
            AnimationPath::Rotation => self.width == 4,
            AnimationPath::Weights => self.width > 0 && self.width <= 10_000,
            _ => self.width == 3,
        };
        ensure(width_ok, "animation component width mismatch")?;
        let expected = self
            .times
            .len()
            .checked_mul(self.width)
            .and_then(|n| n.checked_mul(factor))
            .filter(|&n| n <= 64_000_000)
            .ok_or_else(|| bad("animation values exceed limit"))?;
        ensure(
            self.values.len() == expected && self.values.iter().all(|v| v.is_finite()),
            "invalid animation values/count",
        )?;
        if self.path == AnimationPath::Rotation {
            for key in 0..self.times.len() {
                let offset = (key * factor + usize::from(factor == 3)) * 4;
                let q = &self.values[offset..offset + 4];
                ensure(
                    (q.iter().map(|x| x * x).sum::<f64>() - 1.).abs() < 1e-3,
                    "rotation key must be a unit quaternion",
                )?;
            }
        }
        Ok(())
    }
    /// Clamp before/after the channel's key range. Cubic rotation values are
    /// normalized after component-wise Hermite interpolation, without sign flips.
    pub fn sample(&self, time: f64) -> Result<Vec<f64>> {
        self.validate()?;
        ensure(time.is_finite(), "animation time must be finite")?;
        let factor = if self.interpolation == Interpolation::CubicSpline {
            3
        } else {
            1
        };
        let value = |key| {
            let start = (key * factor + usize::from(factor == 3)) * self.width;
            &self.values[start..start + self.width]
        };
        let mut out = if time <= self.times[0] {
            value(0).to_vec()
        } else if time >= self.times[self.times.len() - 1] {
            value(self.times.len() - 1).to_vec()
        } else {
            let next = self.times.partition_point(|&t| t <= time);
            let prev = next - 1;
            let dt = self.times[next] - self.times[prev];
            let t = (time - self.times[prev]) / dt;
            match self.interpolation {
                Interpolation::Step => value(prev).to_vec(),
                Interpolation::Linear if self.path == AnimationPath::Rotation => {
                    let a = value(prev);
                    let mut b = value(next).to_vec();
                    let mut dot: f64 = a.iter().zip(&b).map(|(a, b)| a * b).sum();
                    if dot < 0. {
                        for x in &mut b {
                            *x = -*x;
                        }
                        dot = -dot;
                    }
                    let (wa, wb) = if dot > 0.9995 {
                        (1. - t, t)
                    } else {
                        let angle = dot.clamp(-1., 1.).acos();
                        (
                            ((1. - t) * angle).sin() / angle.sin(),
                            (t * angle).sin() / angle.sin(),
                        )
                    };
                    a.iter().zip(b).map(|(a, b)| wa * a + wb * b).collect()
                }
                Interpolation::Linear => value(prev)
                    .iter()
                    .zip(value(next))
                    .map(|(a, b)| (1. - t) * a + t * b)
                    .collect(),
                Interpolation::CubicSpline => {
                    let t2 = t * t;
                    let t3 = t2 * t;
                    let out_tangent = (prev * 3 + 2) * self.width;
                    let in_tangent = next * 3 * self.width;
                    (0..self.width)
                        .map(|i| {
                            (2. * t3 - 3. * t2 + 1.) * value(prev)[i]
                                + (t3 - 2. * t2 + t) * dt * self.values[out_tangent + i]
                                + (-2. * t3 + 3. * t2) * value(next)[i]
                                + (t3 - t2) * dt * self.values[in_tangent + i]
                        })
                        .collect()
                }
            }
        };
        if self.path == AnimationPath::Rotation {
            let norm = out.iter().map(|v| v * v).sum::<f64>().sqrt();
            ensure(
                norm.is_finite() && norm > 1e-12,
                "interpolated quaternion is zero/non-finite",
            )?;
            for v in &mut out {
                *v /= norm;
            }
        }
        ensure(
            out.iter().all(|x| x.is_finite()),
            "non-finite animation result",
        )?;
        Ok(out)
    }
}
impl GltfAsset {
    fn validate_channel(&self, c: &AnimationChannel) -> Result<()> {
        c.validate()?;
        let n = self.graph.root["nodes"]
            .as_array()
            .and_then(|a| a.get(c.node))
            .ok_or_else(|| bad("animation node out of range"))?;
        if c.path == AnimationPath::Weights {
            let mi = index(n, "mesh")?;
            let primitives = self.graph.root["meshes"]
                .as_array()
                .and_then(|a| a.get(mi))
                .and_then(|m| m["primitives"].as_array())
                .ok_or_else(|| bad("morph animation requires a mesh"))?;
            ensure(
                !primitives.is_empty()
                    && primitives
                        .iter()
                        .all(|p| p["targets"].as_array().is_some_and(|v| v.len() == c.width)),
                "animation weight width must match all mesh primitives",
            )?;
        } else {
            ensure(
                n.get("matrix").is_none(),
                "TRS animation cannot target a matrix node",
            )?;
        }
        Ok(())
    }
    pub fn animation_clips(&self) -> Result<Vec<AnimationClip>> {
        let Some(animations) = self.graph.root.get("animations") else {
            return Ok(Vec::new());
        };
        let mut clips = Vec::new();
        for (id, animation) in animations
            .as_array()
            .ok_or_else(|| bad("animations must be an array"))?
            .iter()
            .enumerate()
        {
            let mut channels = Vec::new();
            let mut targets = BTreeSet::new();
            let samplers = animation["samplers"]
                .as_array()
                .ok_or_else(|| bad("animation has no samplers"))?;
            for channel in animation["channels"]
                .as_array()
                .ok_or_else(|| bad("animation has no channels"))?
            {
                // Base glTF permits target-node-less extension channels to be ignored.
                if channel["target"].get("node").is_none() {
                    continue;
                }
                let node = index(&channel["target"], "node")?;
                let path: AnimationPath =
                    serde_json::from_value(channel["target"]["path"].clone())?;
                ensure(
                    targets.insert((node, path.key())),
                    "duplicate animation target",
                )?;
                let sampler = samplers
                    .get(index(channel, "sampler")?)
                    .ok_or_else(|| bad("animation sampler out of range"))?;
                let input = index(sampler, "input")?;
                let output = index(sampler, "output")?;
                let accessors = self.graph.root["accessors"]
                    .as_array()
                    .ok_or_else(|| bad("animation accessors must be an array"))?;
                let input_meta = accessors
                    .get(input)
                    .ok_or_else(|| bad("animation input accessor out of range"))?;
                let output_meta = accessors
                    .get(output)
                    .ok_or_else(|| bad("animation output accessor out of range"))?;
                let normalized = |a: &serde_json::Value| -> Result<bool> {
                    a.get("normalized")
                        .map(|v| {
                            v.as_bool()
                                .ok_or_else(|| bad("invalid accessor normalized flag"))
                        })
                        .transpose()
                        .map(|v| v.unwrap_or(false))
                };
                ensure(
                    input_meta["componentType"].as_u64() == Some(5126) && !normalized(input_meta)?,
                    "animation input must be an unnormalized float accessor",
                )?;
                // glTF 2.0 section 3.11: TRS translations/scales require FLOAT;
                // rotations and morph weights also permit normalized 8/16-bit integers.
                let component = output_meta["componentType"].as_u64();
                let output_normalized = normalized(output_meta)?;
                ensure(
                    (component == Some(5126) && !output_normalized)
                        || (matches!(path, AnimationPath::Rotation | AnimationPath::Weights)
                            && matches!(component, Some(5120..=5123))
                            && output_normalized),
                    "unsupported animation output component type/normalization",
                )?;
                let times = self.graph.accessor(input, 1)?;
                for (bound, actual) in [("min", times.first()), ("max", times.last())] {
                    let values = input_meta[bound]
                        .as_array()
                        .ok_or_else(|| bad("animation time accessor needs scalar min/max"))?;
                    ensure(
                        values.len() == 1
                            && values[0].as_f64().map(|v| v as f32 as f64).as_ref() == actual,
                        "animation time bounds must match the first and last keys",
                    )?;
                }
                let width = match path {
                    AnimationPath::Rotation => 4,
                    AnimationPath::Weights => {
                        let n = self.graph.root["nodes"]
                            .as_array()
                            .and_then(|a| a.get(node))
                            .ok_or_else(|| bad("invalid animation node"))?;
                        let m = index(n, "mesh")?;
                        self.graph.root["meshes"][m]["primitives"][0]["targets"]
                            .as_array()
                            .ok_or_else(|| bad("animated mesh has no morphs"))?
                            .len()
                    }
                    _ => 3,
                };
                let values = self.graph.accessor(
                    output,
                    if path == AnimationPath::Weights {
                        1
                    } else {
                        width
                    },
                )?;
                let interpolation = sampler
                    .get("interpolation")
                    .map(|v| serde_json::from_value(v.clone()))
                    .transpose()?
                    .unwrap_or_default();
                let c = AnimationChannel {
                    node,
                    path,
                    interpolation,
                    times,
                    width,
                    values,
                };
                self.validate_channel(&c)?;
                channels.push(c);
            }
            clips.push(AnimationClip {
                name: animation["name"]
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("animation-{id}")),
                channels,
            });
        }
        Ok(clips)
    }
    pub fn geometry_at(&self, clip: usize, time: f64) -> Result<SurfaceMesh> {
        let clips = self.animation_clips()?;
        let clip = clips
            .get(clip)
            .ok_or_else(|| bad("animation index out of range"))?;
        let mut posed = self.clone();
        for channel in &clip.channels {
            posed.graph.root["nodes"][channel.node][channel.path.key()] =
                json!(channel.sample(time)?);
        }
        posed.geometry()
    }
    /// Add a validated native clip. Payloads are stored as f32 glTF accessors.
    pub fn add_animation(&mut self, clip: &AnimationClip) -> Result<usize> {
        ensure(!clip.channels.is_empty(), "animation needs a channel")?;
        let mut targets = BTreeSet::new();
        for c in &clip.channels {
            self.validate_channel(c)?;
            ensure(
                targets.insert((c.node, c.path.key())),
                "duplicate animation target",
            )?;
            ensure(
                c.times.windows(2).all(|w| (w[1] as f32) > (w[0] as f32)),
                "animation times collapse in f32",
            )?;
        }
        let mut next = self.clone();
        let mut samplers = Vec::new();
        let mut channels = Vec::new();
        for c in &clip.channels {
            let input = next.append_floats(&c.times, 1, "SCALAR", true)?;
            let (width, ty) = match c.path {
                AnimationPath::Weights => (1, "SCALAR"),
                AnimationPath::Rotation => (4, "VEC4"),
                _ => (3, "VEC3"),
            };
            let output = next.append_floats(&c.values, width, ty, false)?;
            channels.push(json!({"sampler":samplers.len(),"target":{"node":c.node,"path":c.path}}));
            samplers.push(json!({"input":input,"output":output,"interpolation":c.interpolation}));
        }
        let animations = array_mut(&mut next.graph.root, "animations")?;
        let id = animations.len();
        animations.push(json!({"name":clip.name,"channels":channels,"samplers":samplers}));
        next.animation_clips()?;
        *self = next;
        Ok(id)
    }
    pub(crate) fn append_floats(
        &mut self,
        values: &[f64],
        width: usize,
        ty: &str,
        bounds: bool,
    ) -> Result<usize> {
        ensure(
            !values.is_empty() && values.len().is_multiple_of(width),
            "invalid accessor length",
        )?;
        let floats: Vec<f32> = values.iter().map(|v| *v as f32).collect();
        ensure(
            floats.iter().all(|x| x.is_finite()),
            "value outside glTF f32 range",
        )?;
        let bytes: Vec<_> = floats.iter().flat_map(|x| x.to_le_bytes()).collect();
        let view = self.append_bytes(bytes)?;
        let mut a =
            json!({"bufferView":view,"componentType":5126,"count":values.len()/width,"type":ty});
        if bounds {
            let mut low = vec![f32::INFINITY; width];
            let mut high = vec![f32::NEG_INFINITY; width];
            for row in floats.chunks_exact(width) {
                for i in 0..width {
                    low[i] = low[i].min(row[i]);
                    high[i] = high[i].max(row[i]);
                }
            }
            a["min"] = json!(low);
            a["max"] = json!(high);
        }
        let a_list = array_mut(&mut self.graph.root, "accessors")?;
        let id = a_list.len();
        a_list.push(a);
        Ok(id)
    }
    pub(crate) fn append_bytes(&mut self, bytes: Vec<u8>) -> Result<usize> {
        let total: usize = self.graph.buffers.iter().map(Vec::len).sum();
        ensure(
            !bytes.is_empty() && bytes.len() <= super::LIMIT.saturating_sub(total),
            "added buffer exceeds byte limit",
        )?;
        let id = self.graph.buffers.len();
        let n = bytes.len();
        self.graph.buffers.push(bytes);
        array_mut(&mut self.graph.root, "buffers")?.push(json!({"byteLength":n}));
        let views = array_mut(&mut self.graph.root, "bufferViews")?;
        let view = views.len();
        views.push(json!({"buffer":id,"byteLength":n}));
        Ok(view)
    }
}
