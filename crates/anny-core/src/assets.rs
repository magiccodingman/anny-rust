// Port of NAVER Anny source-asset construction and rig transforms.
// Copyright (C) 2025 NAVER Corp. SPDX-License-Identifier: Apache-2.0
use crate::{
    config::*,
    ensure,
    math::*,
    mesh,
    model::{Anny, ModelData, ModelMetadata},
    tensor::{Archive, Kind},
    Error, Result, Tensor,
};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{BufRead, BufReader, Read},
    path::{Path, PathBuf},
};

#[derive(Clone, Debug)]
pub struct AssetStore {
    pub root: PathBuf,
}
impl AssetStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }
    pub fn build(&self, config: &AnnyConfig) -> Result<Anny> {
        config.validate()?;
        let rig = config.rig.resolve()?;
        let topology = config.topology.resolve()?;
        let d = if rig.base_rig == "soma" {
            self.build_soma(config, &topology)?
        } else if topology.base_mesh == "makehuman" {
            self.build_native(config, &rig, &topology)?
        } else {
            self.build_alternative(config, &rig, &topology)?
        };
        Anny::from_model_data(d, config.clone())
    }
    pub fn converted(&self, name: &str) -> Result<Archive> {
        let path = self.root.join(format!("{name}.safetensors"));
        if path.is_file() {
            Archive::load(path)
        } else {
            // A raw pinned upstream checkout is supported without a preparation
            // interpreter. Prepared portable siblings remain the fast path.
            crate::torch_archive::load(self.root.join(name))
        }
    }
    fn json(&self, name: impl AsRef<Path>) -> Result<Value> {
        let path = self.root.join(name);
        if path.is_file() {
            return Ok(serde_json::from_slice(&std::fs::read(path)?)?);
        }
        // The portable YAML sibling is optional for untouched upstream trees.
        let original = path.with_extension("");
        if matches!(
            original.extension().and_then(|s| s.to_str()),
            Some("yaml" | "yml")
        ) {
            return serde_yaml::from_slice(&std::fs::read(original)?)
                .map_err(|e| Error::Invalid(format!("invalid asset YAML: {e}")));
        }
        Err(Error::Invalid(format!("missing asset {}", path.display())))
    }
    pub fn build_native(
        &self,
        config: &AnnyConfig,
        rig: &RigConfig,
        top: &TopologyConfig,
    ) -> Result<ModelData> {
        let mut d = self.load_raw(config, rig, top)?;
        filter_rig(&mut d, &rig.bones_to_remove, rig.subtree_root.as_deref())?;
        if top.nudity_edits || top.submodel != "body" {
            mesh::edit_mesh(&mut d)?;
        }
        if top.submodel != "body" {
            let labels: Vec<&str> = match top.submodel.as_str() {
                "head" => vec![
                    "head",
                    "eye_cavity.R",
                    "eye_cavity.L",
                    "mouth_cavity",
                    "eye_front.L",
                    "eye_back.L",
                    "eye_front.R",
                    "eye_back.L",
                    "tongue",
                ],
                "hand.L" => vec!["hand.L"],
                "hand.R" => vec!["hand.R"],
                _ => return Err(Error::Invalid("unknown submodel".into())),
            };
            let keep = self.segment_faces(&d, &labels)?;
            mesh::filter_faces(&mut d, &keep)?;
        }
        if top.remove_unattached_vertices {
            mesh::remove_unattached_vertices(&mut d)?;
        }
        mesh::compact_skinning_weights(&mut d)?;
        if top.triangulate_faces {
            mesh::triangulate(&mut d)?;
        }
        self.apply_orientation(&mut d, rig)?;
        Ok(d)
    }
    pub fn apply_orientation(&self, d: &mut ModelData, rig: &RigConfig) -> Result<()> {
        match rig.bone_orientation {
            BoneOrientation::Blender => Ok(()),
            BoneOrientation::Cached => self.cached_orientation(d, rig.subtree_root.as_deref()),
            BoneOrientation::Procrustes => procrustes_orientation(d),
        }
    }
    fn load_raw(
        &self,
        config: &AnnyConfig,
        rig: &RigConfig,
        top: &TopologyConfig,
    ) -> Result<ModelData> {
        let obj = mesh::load_obj(self.root.join("mpfb2/3dobjs/base.obj"))?;
        let n = obj.vertices.len();
        let vertices = Tensor::new(
            vec![n, 3],
            obj.vertices
                .iter()
                .flat_map(|v| world(*v, 0.1).as_slice().to_vec())
                .collect(),
        )?;
        let mut groups = vec!["body"];
        if top.eyes {
            groups.extend(["helper-l-eye", "helper-r-eye"]);
        }
        if top.tongue {
            groups.push("helper-tongue");
        }
        let (mut faces, mut uv_faces) = (Vec::new(), Vec::new());
        for name in groups {
            let g = obj
                .groups
                .get(name)
                .ok_or_else(|| Error::Invalid(format!("missing OBJ group {name}")))?;
            faces.extend(g.faces.iter().cloned());
            uv_faces.extend(g.uv_faces.iter().cloned());
        }
        let (blendshapes, mask, labels) = self.load_blendshapes(config, &vertices)?;
        let c = labels.len();
        let (rigfile, weightfile) = rig.preset_files()?;
        let rigvalue = self.json(rigfile)?;
        let rigdata = rigvalue
            .get("bones")
            .unwrap_or(&rigvalue)
            .as_object()
            .ok_or_else(|| Error::Invalid("rig must be a dictionary".into()))?;
        let roots: Vec<_> = rigdata
            .iter()
            .filter(|(_, v)| {
                v.get("parent")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .is_empty()
            })
            .map(|(k, _)| k.clone())
            .collect();
        ensure(roots.len() == 1, "rig requires one root")?;
        let (mut bone_labels, mut bone_parents) = (Vec::new(), Vec::new());
        fn walk(
            name: &str,
            parent: i32,
            rd: &serde_json::Map<String, Value>,
            labels: &mut Vec<String>,
            parents: &mut Vec<i32>,
        ) -> Result<()> {
            ensure(!labels.iter().any(|s| s == name), "cycle in rig")?;
            let i = labels.len();
            labels.push(name.into());
            parents.push(parent);
            for (child, node) in rd {
                if node.get("parent").and_then(Value::as_str) == Some(name) {
                    walk(child, i as i32, rd, labels, parents)?;
                }
            }
            Ok(())
        }
        walk(&roots[0], -1, rigdata, &mut bone_labels, &mut bone_parents)?;
        ensure(bone_labels.len() == rigdata.len(), "unreachable rig bones")?;
        let j = bone_labels.len();
        let mut heads = Tensor::zeros(vec![j, 3]);
        let mut tails = heads.clone();
        let mut head_shapes = Tensor::zeros(vec![c, j, 3]);
        let mut tail_shapes = head_shapes.clone();
        let mut rolls = Tensor::zeros(vec![1, j, 3, 3]);
        for (i, label) in bone_labels.iter().enumerate() {
            let node = &rigdata[label];
            let head = coordinates(&obj, &node["head"])?;
            let tail = coordinates(&obj, &node["tail"])?;
            for (indices, out, shapes) in [
                (&head, &mut heads, &mut head_shapes),
                (&tail, &mut tails, &mut tail_shapes),
            ] {
                ensure(
                    !indices.is_empty() && indices.iter().all(|&v| v < n),
                    "invalid bone coordinate regressor",
                )?;
                for &v in indices {
                    for axis in 0..3 {
                        out.data[i * 3 + axis] +=
                            vertices.data[v * 3 + axis] / indices.len() as f64;
                    }
                }
                for b in 0..c {
                    for &v in indices {
                        for axis in 0..3 {
                            shapes.data[(b * j + i) * 3 + axis] +=
                                blendshapes.data[(b * n + v) * 3 + axis] / indices.len() as f64;
                        }
                    }
                }
            }
            let roll = node
                .get("roll")
                .and_then(Value::as_f64)
                .ok_or_else(|| Error::Invalid(format!("bone {label} lacks roll")))?;
            write3(&legacy_roll_y(roll), &mut rolls.data[i * 9..(i + 1) * 9]);
        }
        let weights_value = self.json(weightfile)?;
        let weights = weights_value["weights"]
            .as_object()
            .ok_or_else(|| Error::Invalid("missing weights dictionary".into()))?;
        let mut per_vertex = vec![Vec::<(usize, f64)>::new(); n];
        for (i, name) in bone_labels.iter().enumerate() {
            if let Some(values) = weights.get(name) {
                for entry in values
                    .as_array()
                    .ok_or_else(|| Error::Invalid("weights must be arrays".into()))?
                {
                    let v = entry[0]
                        .as_u64()
                        .ok_or_else(|| Error::Invalid("invalid vertex id in weights".into()))?
                        as usize;
                    let w = entry[1]
                        .as_f64()
                        .ok_or_else(|| Error::Invalid("invalid weight".into()))?;
                    ensure(
                        v < n && w.is_finite() && w >= 0.,
                        "invalid skinning influence",
                    )?;
                    per_vertex[v].push((i, w));
                }
            }
        }
        let k = per_vertex.iter().map(Vec::len).max().unwrap_or(1).max(1);
        let mut weights = Tensor::zeros(vec![n, k]);
        let mut indices = weights.clone();
        indices.kind = Kind::Index;
        for (v, entries) in per_vertex.iter().enumerate() {
            let sum: f64 = entries.iter().map(|x| x.1).sum();
            // Helpers with no weights are discarded by normal topologies. Bind them to root
            // rather than manufacture NaNs in the portable full-vertex representation.
            if sum == 0. {
                weights.data[v * k] = 1.;
                continue;
            }
            for (s, &(bone, w)) in entries.iter().enumerate() {
                indices.data[v * k + s] = bone as f64;
                weights.data[v * k + s] = w / sum;
            }
        }
        let mut d = ModelData {
            metadata: ModelMetadata {
                bone_labels,
                bone_parents,
                blendshape_labels: labels,
            },
            arrays: BTreeMap::new(),
        };
        for (name, t) in [
            ("template_vertices", vertices),
            ("blendshapes", blendshapes),
            ("stacked_phenotype_blend_shapes_mask", mask),
            ("faces", mesh::pack_faces(&faces)?),
            (
                "face_texture_coordinate_indices",
                mesh::pack_faces(&uv_faces)?,
            ),
            (
                "texture_coordinates",
                Tensor::new(
                    vec![obj.uv.len(), 2],
                    obj.uv.iter().flatten().copied().collect(),
                )?,
            ),
            ("template_bone_heads", heads),
            ("template_bone_tails", tails),
            ("bone_heads_blendshapes", head_shapes),
            ("bone_tails_blendshapes", tail_shapes),
            ("bone_rolls_rotmat", rolls),
            ("vertex_bone_weights", weights),
            ("vertex_bone_indices", indices),
            (
                "base_mesh_vertex_indices",
                Tensor::indices(vec![n], (0..n).collect()),
            ),
        ] {
            d.put(name, t);
        }
        Ok(d)
    }
    fn load_blendshapes(
        &self,
        config: &AnnyConfig,
        vertices: &Tensor,
    ) -> Result<(Tensor, Tensor, Vec<String>)> {
        let n = vertices.shape[0];
        let (mut shapes, mut masks, mut labels) = (Vec::new(), Vec::new(), Vec::new());
        let components: Vec<_> = PHENOTYPE_VARIATIONS
            .iter()
            .flat_map(|(_, v)| v.iter().copied())
            .collect();
        for (block, keys) in [
            ("universal", vec!["gender", "age", "muscle", "weight"]),
            ("race", vec!["race", "gender", "age"]),
            (
                "height",
                vec!["gender", "age", "muscle", "weight", "height"],
            ),
            (
                "proportions",
                vec!["gender", "age", "muscle", "weight", "proportions"],
            ),
            (
                "breast",
                vec!["gender", "age", "muscle", "weight", "cupsize", "firmness"],
            ),
        ] {
            for values in combinations(&keys) {
                if block == "breast" && values[0] != "female" {
                    continue;
                }
                if block == "proportions" && values.iter().any(|s| s == "newborn" || s == "baby") {
                    continue;
                }
                let newborn = values.iter().any(|s| s == "newborn");
                let actual: Vec<_> = values
                    .iter()
                    .map(|v| {
                        if v == "newborn" && block != "breast" {
                            "baby"
                        } else {
                            v.as_str()
                        }
                    })
                    .collect();
                let basename = actual.join("-");
                let relative = match block {
                    "universal" => format!("macrodetails/universal-{basename}.target.gz"),
                    "race" => format!("macrodetails/{basename}.target.gz"),
                    "height" | "proportions" => {
                        format!("macrodetails/{block}/{basename}.target.gz")
                    }
                    _ => format!("breast/{basename}.target.gz"),
                };
                let file = self.root.join("mpfb2/targets").join(relative);
                if block == "breast" && !file.is_file() {
                    continue;
                }
                let mut shape = load_target(&file, n)?;
                if newborn {
                    let scale = [0.922, 0.922, 0.75];
                    for (i, v) in shape.iter_mut().enumerate() {
                        *v = scale[i % 3] * *v + (scale[i % 3] - 1.) / 3. * vertices.data[i];
                    }
                }
                shapes.extend(shape);
                labels.push(format!("{block}:{}", values.join("-")));
                masks.extend(components.iter().map(|p| {
                    if values.iter().any(|v| v == p) {
                        1.
                    } else {
                        0.
                    }
                }));
            }
        }
        let macro_count = labels.len();
        let facial: Vec<String> = FACIAL_ACTION_LABELS.iter().map(|s| (*s).into()).collect();
        let selected = config.facial_actions.mask(&facial, false)?;
        for (i, name) in facial.iter().enumerate() {
            if selected[i] {
                shapes.extend(load_target(
                    &self
                        .root
                        .join(format!("faceunits01/targets/faceunits/{name}.target")),
                    n,
                )?);
                labels.push(format!("facial_action:{name}"));
            }
        }
        let metadata = self.json("mpfb2/targets/target.json")?;
        let mut locals = Vec::new();
        for (key, meta) in metadata
            .as_object()
            .ok_or_else(|| Error::Invalid("invalid target metadata".into()))?
        {
            if key == "genitals" {
                continue;
            }
            for category in meta["categories"]
                .as_array()
                .ok_or_else(|| Error::Invalid("missing target categories".into()))?
            {
                if let Some(o) = category.get("opposites") {
                    for side in ["left", "right", "unsided"] {
                        let neg = o[format!("negative-{side}")].as_str().unwrap_or("");
                        let pos = o[format!("positive-{side}")].as_str().unwrap_or("");
                        if !neg.is_empty() && !pos.is_empty() {
                            locals.push((key.clone(), pos.to_string(), neg.to_string()));
                        }
                    }
                }
            }
        }
        let names = locals.iter().map(|x| x.1.clone()).collect::<Vec<_>>();
        let selected = config.local_changes.mask(&names, true)?;
        for (i, (category, pos, neg)) in locals.iter().enumerate() {
            if selected[i] {
                for name in [pos, neg] {
                    shapes.extend(load_target(
                        &self
                            .root
                            .join(format!("mpfb2/targets/{category}/{name}.target.gz")),
                        n,
                    )?);
                    labels.push(format!("local_change:{name}"));
                }
            }
        }
        Ok((
            Tensor::new(vec![labels.len(), n, 3], shapes)?,
            Tensor::new(vec![macro_count, 26], masks)?,
            labels,
        ))
    }
    pub fn cached_orientation(&self, d: &mut ModelData, root_label: Option<&str>) -> Result<()> {
        let a = self.converted("cached/anny.pth")?;
        let payload = a.payload()?;
        let source: Vec<String> = serde_json::from_value(payload["bone_labels"].clone())?;
        let source_shapes: Vec<String> =
            serde_json::from_value(payload["blendshape_labels"].clone())?;
        let mut select = Vec::new();
        for (i, label) in d.metadata.bone_labels.iter().enumerate() {
            let l = if i == 0 {
                root_label.unwrap_or(label)
            } else {
                label
            };
            select.push(source.iter().position(|s| s == l).ok_or_else(|| {
                Error::Invalid(format!("cached Anny orientations do not cover bone {l}"))
            })?);
        }
        for name in [
            "bone_template_orientation_matrices",
            "reference_bone_orientations",
            "template_bone_heads",
        ] {
            d.put(name, a.payload_tensor(name)?.select(0, &select)?);
        }
        for name in ["bone_orientation_blendshapes", "bone_heads_blendshapes"] {
            let t = a.payload_tensor(name)?.select(1, &select)?;
            let t = select_blendshape_rows(&t, &source_shapes, &d.metadata.blendshape_labels)?;
            d.put(name, t);
        }
        d.remove(&[
            "template_bone_tails",
            "bone_tails_blendshapes",
            "bone_rolls_rotmat",
            "bone_nonzeroweight_mask",
            "bone_vertex_indices",
            "bone_vertex_weights",
            "template_bone_vertices",
            "bone_children_indices",
            "bone_children_mask",
            "bone_children_local_offsets",
        ]);
        Ok(())
    }
    pub fn segment_faces(&self, d: &ModelData, labels: &[&str]) -> Result<Vec<usize>> {
        let meta = self.json("segmentation/body_parts_segmentation.yaml.json")?;
        let colors: Vec<[u8; 3]> = labels
            .iter()
            .map(|s| serde_json::from_value(meta["colors"][*s].clone()).map_err(Error::from))
            .collect::<Result<_>>()?;
        let file = std::fs::File::open(self.root.join("segmentation/body_parts_segmentation.png"))?;
        let mut decoder = png::Decoder::new(BufReader::new(file));
        decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
        let mut reader = decoder
            .read_info()
            .map_err(|e| Error::Invalid(e.to_string()))?;
        let mut buf = vec![0; reader.output_buffer_size()];
        let info = reader
            .next_frame(&mut buf)
            .map_err(|e| Error::Invalid(e.to_string()))?;
        let channels = match info.color_type {
            png::ColorType::Rgb => 3,
            png::ColorType::Rgba => 4,
            _ => return Err(Error::Invalid("segmentation image must be RGB(A)".into())),
        };
        let uv = d.get("texture_coordinates")?;
        let ft = d.get("face_texture_coordinate_indices")?;
        let h = info.height as usize;
        let w = info.width as usize;
        let mut keep = Vec::new();
        for (face, f) in ft.data.chunks_exact(ft.shape[1]).enumerate() {
            let mut p = [0.; 2];
            for &i in f {
                p[0] += uv.data[i as usize * 2] / f.len() as f64;
                p[1] += uv.data[i as usize * 2 + 1] / f.len() as f64;
            }
            let x = (p[0] * w as f64)
                .round_ties_even()
                .clamp(0., (w - 1) as f64) as usize;
            let y = ((1. - p[1]) * h as f64)
                .round_ties_even()
                .clamp(0., (h - 1) as f64) as usize;
            let pixel = &buf[(y * w + x) * channels..(y * w + x) * channels + 3];
            if colors.iter().any(|c| c.as_slice() == pixel) {
                keep.push(face);
            }
        }
        Ok(keep)
    }
    fn target_mesh(&self, name: &str) -> Result<(Tensor, Tensor)> {
        let file = match name {
            "soma" => "soma/SOMA_wrap.obj".into(),
            "anny_from_soma" => "soma/base_body.obj".into(),
            _ => format!("topology/{name}.obj"),
        };
        let obj = mesh::load_obj(self.root.join(file))?;
        let vertices = Tensor::new(
            vec![obj.vertices.len(), 3],
            obj.vertices
                .iter()
                .flat_map(|v| world(*v, 1.).as_slice().to_vec())
                .collect(),
        )?;
        let g = obj
            .groups
            .get("noname")
            .ok_or_else(|| Error::Invalid("topology OBJ missing noname group".into()))?;
        let faces = if g.faces.iter().all(|f| f.len() == g.faces[0].len()) {
            mesh::pack_faces(&g.faces)?
        } else {
            let mut triangles = Vec::new();
            for f in &g.faces {
                if f.len() == 3 {
                    triangles.push(f.clone());
                } else {
                    let point = |i: usize| vec3(&vertices.data[f[i] * 3..f[i] * 3 + 3]);
                    let split = if (point(0) - point(2)).norm() < (point(1) - point(3)).norm() {
                        [[0, 1, 2], [2, 3, 0]]
                    } else {
                        [[0, 1, 3], [3, 1, 2]]
                    };
                    for tri in split {
                        triangles.push(tri.iter().map(|&i| f[i]).collect());
                    }
                }
            }
            mesh::pack_faces(&triangles)?
        };
        Ok((vertices, faces))
    }
    pub fn build_alternative(
        &self,
        config: &AnnyConfig,
        rig: &RigConfig,
        top: &TopologyConfig,
    ) -> Result<ModelData> {
        let mut source_rig = rig.clone();
        source_rig.bone_orientation = BoneOrientation::Blender;
        let source_top = if ["smpl", "smplx"].contains(&top.base_mesh.as_str()) {
            TopologyConfig {
                nudity_edits: false,
                tongue: false,
                remove_unattached_vertices: false,
                ..TopologyConfig::default()
            }
        } else {
            TopologyConfig::default()
        };
        let mut d = self.build_native(config, &source_rig, &source_top)?;
        if ["smpl", "smplx"].contains(&top.base_mesh.as_str()) {
            let a = self.converted(&format!("noncommercial/anny2{}.pth", top.base_mesh))?;
            let ids = a.payload_tensor("anny2dst_vertex_indices")?;
            let weights = payload_barycentric(
                &a,
                "anny2dst_barycentric_coordinates",
                ids.shape[0],
                ids.shape[1],
            )?;
            let faces = a.payload_tensor("dst_faces")?.clone();
            interpolate_model_data(&mut d, ids, &weights, faces, None, None, true)?;
        } else {
            let (source_vertices, source_faces) = if top.base_mesh == "soma" {
                self.target_mesh("anny_from_soma")?
            } else {
                (d.get("template_vertices")?.clone(), d.get("faces")?.clone())
            };
            let (target_vertices, target_faces) = self.target_mesh(&top.base_mesh)?;
            apply_retopology_from_mesh(
                &mut d,
                &target_vertices,
                target_faces,
                &source_vertices,
                &source_faces,
            )?;
        }
        self.apply_orientation(&mut d, rig)?;
        Ok(d)
    }
    pub fn build_soma(&self, config: &AnnyConfig, top: &TopologyConfig) -> Result<ModelData> {
        let mut d = self.build_alternative(
            config,
            &RigConfig::parse("anny")?,
            &TopologyConfig::parse("soma")?,
        )?;
        let a = self.converted("soma/soma_rig.pt")?;
        let cov = self.converted("cached/soma.pth")?;
        apply_soma_rig(&mut d, &a, &cov)?;
        let faces = self.converted("soma/soma_faces.pt")?;
        let payload = faces.payload()?;
        let name = payload["__tensor__"]
            .as_str()
            .ok_or_else(|| Error::Invalid("invalid soma faces".into()))?;
        d.put("faces", faces.tensor(name)?.clone());
        if top.base_mesh != "soma" {
            let target = if top.base_mesh == "makehuman" {
                self.build_native(config, &RigConfig::parse("anny")?, top)?
            } else {
                self.build_alternative(config, &RigConfig::parse("anny")?, top)?
            };
            let source_vertices = d.get("template_vertices")?.clone();
            let source_faces = d.get("faces")?.clone();
            let vertices = target.get("template_vertices")?.clone();
            let (ids, weights) =
                projection_coordinates(&vertices, &source_vertices, &source_faces, None)?;
            interpolate_model_data(
                &mut d,
                &ids,
                &weights,
                target.get("faces")?.clone(),
                Some(vertices),
                Some(target.get("base_mesh_vertex_indices")?.clone()),
                true,
            )?;
        }
        Ok(d)
    }
}
fn world(v: Vec3, scale: f64) -> Vec3 {
    let c = 2.220446049250313e-16;
    Vec3::new(
        scale * v[0],
        scale * (c * v[1] - v[2]),
        scale * (v[1] + c * v[2]),
    )
}
fn load_target(path: &Path, n: usize) -> Result<Vec<f64>> {
    let file = std::fs::File::open(path)?;
    let reader: Box<dyn Read> = if path.extension().is_some_and(|x| x == "gz") {
        Box::new(flate2::read::GzDecoder::new(file))
    } else {
        Box::new(file)
    };
    let mut out = vec![0.; n * 3];
    for (line_id, line) in BufReader::new(reader).lines().enumerate() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<_> = line.split_whitespace().collect();
        ensure(
            parts.len() == 4,
            format!("{}:{} invalid target row", path.display(), line_id + 1),
        )?;
        let i = parts[0]
            .parse::<usize>()
            .map_err(|_| Error::Invalid("invalid target index".into()))?;
        ensure(i < n, "target index out of bounds")?;
        let values = parts[1..]
            .iter()
            .map(|s| {
                s.parse::<f64>()
                    .map_err(|_| Error::Invalid("invalid target coordinate".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        let p = world(vec3(&values), 0.1);
        out[i * 3..i * 3 + 3].copy_from_slice(p.as_slice());
    }
    Ok(out)
}
fn combinations(keys: &[&str]) -> Vec<Vec<String>> {
    let mut result = vec![Vec::new()];
    for key in keys {
        let choices = PHENOTYPE_VARIATIONS
            .iter()
            .find(|(s, _)| s == key)
            .unwrap()
            .1;
        result = result
            .into_iter()
            .flat_map(|prefix| {
                choices.iter().map(move |s| {
                    let mut v = prefix.clone();
                    v.push((*s).into());
                    v
                })
            })
            .collect();
    }
    result
}
fn coordinates(obj: &mesh::Obj, node: &Value) -> Result<Vec<usize>> {
    match node["strategy"].as_str() {
        Some("VERTEX") => Ok(vec![node["vertex_index"]
            .as_u64()
            .ok_or_else(|| Error::Invalid("invalid VERTEX regressor".into()))?
            as usize]),
        Some("MEAN") => Ok(serde_json::from_value(node["vertex_indices"].clone())?),
        Some("CUBE") => {
            let group = node["cube_name"]
                .as_str()
                .ok_or_else(|| Error::Invalid("missing cube name".into()))?;
            let g = obj
                .groups
                .get(group)
                .ok_or_else(|| Error::Invalid(format!("missing joint cube {group}")))?;
            Ok(g.faces
                .iter()
                .flatten()
                .copied()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect())
        }
        _ => Err(Error::Invalid("unknown joint coordinate strategy".into())),
    }
}
pub fn select_blendshape_rows(t: &Tensor, source: &[String], target: &[String]) -> Result<Tensor> {
    ensure(
        !t.shape.is_empty() && t.shape[0] == source.len(),
        "blendshape label/tensor mismatch",
    )?;
    let size: usize = t.shape[1..].iter().product();
    let lookup: BTreeMap<_, _> = source.iter().enumerate().map(|(i, s)| (s, i)).collect();
    let mut data = Vec::with_capacity(target.len() * size);
    for name in target {
        if let Some(&i) = lookup.get(name) {
            data.extend_from_slice(&t.data[i * size..(i + 1) * size]);
        } else if name.starts_with("facial_action:") {
            data.resize(data.len() + size, 0.);
        } else {
            return Err(Error::Invalid(format!(
                "orientation data missing blendshape {name}"
            )));
        }
    }
    let mut shape = t.shape.clone();
    shape[0] = target.len();
    Ok(Tensor {
        shape,
        data,
        kind: t.kind,
    })
}
pub fn filter_rig(
    d: &mut ModelData,
    remove: &BTreeSet<String>,
    subtree: Option<&str>,
) -> Result<()> {
    if remove.is_empty() && subtree.is_none() {
        return Ok(());
    }
    let parents = &d.metadata.bone_parents;
    let labels = &d.metadata.bone_labels;
    let j = labels.len();
    let root = subtree
        .map(|s| {
            labels
                .iter()
                .position(|x| x == s)
                .ok_or_else(|| Error::Invalid(format!("unknown subtree {s}")))
        })
        .transpose()?;
    let mut candidates = vec![root.is_none(); j];
    if let Some(root) = root {
        for i in propagation_order(parents)? {
            candidates[i] = i == root || (parents[i] >= 0 && candidates[parents[i] as usize]);
        }
    }
    let retained: Vec<_> = (0..j)
        .filter(|&i| candidates[i] && !remove.contains(&labels[i]))
        .collect();
    ensure(!retained.is_empty(), "rig filtering removed every bone")?;
    let offset = usize::from(root.is_some());
    let mut map = vec![None; j];
    for (i, &old) in retained.iter().enumerate() {
        map[old] = Some(i + offset);
    }
    let nearest = |mut i: i32| -> Option<usize> {
        while i >= 0 {
            if let Some(new) = map[i as usize] {
                return Some(new);
            }
            i = parents[i as usize];
        }
        None
    };
    let fallback = if root.is_some() {
        0
    } else {
        let roots: Vec<_> = retained
            .iter()
            .filter(|&&i| nearest(parents[i]).is_none())
            .copied()
            .collect();
        ensure(
            roots.len() == 1,
            "filtered rig does not have exactly one root",
        )?;
        map[roots[0]].unwrap()
    };
    let target: Vec<_> = (0..j)
        .map(|i| nearest(i as i32).unwrap_or(fallback))
        .collect();
    let out_j = retained.len() + offset;
    let n = d.vertex_count();
    let old_w = d.get("vertex_bone_weights")?;
    let old_ids = d.get("vertex_bone_indices")?;
    let k = old_w.shape[1];
    let mut rows = Vec::with_capacity(n);
    let mut max = 1;
    for v in 0..n {
        let mut row = vec![0.; out_j];
        for s in 0..k {
            row[target[old_ids.data[v * k + s] as usize]] += old_w.data[v * k + s];
        }
        let entries: Vec<_> = row
            .into_iter()
            .enumerate()
            .filter(|(_, w)| *w > 0.)
            .collect();
        max = max.max(entries.len());
        rows.push(entries);
    }
    let mut ws = Tensor::zeros(vec![n, max]);
    let mut ids = ws.clone();
    ids.kind = Kind::Index;
    for (v, row) in rows.iter().enumerate() {
        for (s, &(bone, w)) in row.iter().enumerate() {
            ws.data[v * max + s] = w;
            ids.data[v * max + s] = bone as f64;
        }
    }
    let mut new_labels: Vec<_> = retained.iter().map(|&i| labels[i].clone()).collect();
    let mut new_parents: Vec<_> = retained
        .iter()
        .map(|&i| {
            nearest(parents[i]).map_or(if root.is_some() { fallback as i32 } else { -1 }, |x| {
                x as i32
            })
        })
        .collect();
    let mut selection = retained;
    if let Some(root) = root {
        selection.insert(0, root);
        new_labels.insert(0, "root".into());
        new_parents.insert(0, -1);
    }
    for (name, axis) in [
        ("template_bone_heads", 0),
        ("template_bone_tails", 0),
        ("bone_heads_blendshapes", 1),
        ("bone_tails_blendshapes", 1),
        ("bone_rolls_rotmat", 1),
    ] {
        if let Some(t) = d.arrays.get(name) {
            let t = t.select(axis, &selection)?;
            d.put(name, t);
        }
    }
    d.metadata.bone_labels = new_labels;
    d.metadata.bone_parents = new_parents;
    d.put("vertex_bone_weights", ws);
    d.put("vertex_bone_indices", ids);
    Ok(())
}
pub fn procrustes_orientation(d: &mut ModelData) -> Result<()> {
    let rig = RigConfig {
        bone_orientation: BoneOrientation::Blender,
        root_identity_orientation: true,
        ..Default::default()
    };
    let rest = crate::model::rest_model(d, &rig, &Tensor::zeros(vec![1, d.blendshape_count()]))?;
    let poses = rest.get("rest_bone_poses")?;
    let n = d.vertex_count();
    let j = d.bone_count();
    let w = d.get("vertex_bone_weights")?;
    let ids = d.get("vertex_bone_indices")?;
    let k = w.shape[1];
    let mut samples = vec![BTreeMap::<usize, f64>::new(); j];
    for v in 0..n {
        for s in 0..k {
            let weight = w.data[v * k + s];
            if weight > 0. {
                *samples[ids.data[v * k + s] as usize].entry(v).or_default() += weight;
            }
        }
    }
    let mut mask = Tensor::zeros(vec![j]);
    mask.kind = Kind::Bool;
    let active: Vec<_> = (0..j).filter(|&i| !samples[i].is_empty()).collect();
    ensure(!active.is_empty(), "no weighted bones")?;
    let max = active.iter().map(|&i| samples[i].len()).max().unwrap();
    let mut inds = Tensor::zeros(vec![active.len(), max]);
    inds.kind = Kind::Index;
    let mut weights = Tensor::zeros(vec![active.len(), max]);
    let mut vertices = Tensor::zeros(vec![active.len(), max, 3]);
    let tv = d.get("template_vertices")?;
    for (row, &bone) in active.iter().enumerate() {
        mask.data[bone] = 1.;
        let inv = inverse_rigid(&mat4(&poses.data[bone * 16..(bone + 1) * 16]));
        let sum: f64 = samples[bone].values().sum();
        for s in 0..max {
            let (&v, &w) = samples[bone].iter().nth(s).unwrap_or((&0, &0.));
            inds.data[row * max + s] = v as f64;
            weights.data[row * max + s] = w / sum;
            let p = point(&inv, &vec3(&tv.data[v * 3..v * 3 + 3]));
            vertices.data[(row * max + s) * 3..(row * max + s + 1) * 3]
                .copy_from_slice(p.as_slice());
        }
    }
    d.put("bone_nonzeroweight_mask", mask);
    d.put("bone_vertex_indices", inds);
    d.put("bone_vertex_weights", weights);
    d.put("template_bone_vertices", vertices);
    d.remove(&[
        "template_bone_tails",
        "bone_tails_blendshapes",
        "bone_rolls_rotmat",
    ]);
    Ok(())
}

/// Resample mesh and morphs while carrying topology-independent skeleton data.
/// `indices` and `weights` are N x K, not transposed. Arbitrary signed weights
/// can be used with `check_convex=false`; final bone weights are normalized.
#[allow(clippy::too_many_arguments)]
pub fn interpolate_model_data(
    d: &mut ModelData,
    indices: &Tensor,
    weights: &Tensor,
    faces: Tensor,
    vertices: Option<Tensor>,
    base_indices: Option<Tensor>,
    check_convex: bool,
) -> Result<()> {
    let old_n = d.vertex_count();
    ensure(
        indices.shape.len() == 2 && indices.shape[0] > 0 && indices.shape[1] > 0,
        "resampling needs N x K indices",
    )?;
    weights.expect_shape(&indices.shape, "resampling weights")?;
    let ids = indices.checked_indices(old_n, "resampling indices")?;
    if !d.arrays.contains_key("bone_template_orientation_matrices") {
        ensure(!d.arrays.contains_key("bone_vertex_indices"),"point resampling cannot carry vertex-indexed runtime Procrustes buffers; use cached orientations")?;
    }
    let n = indices.shape[0];
    let k = indices.shape[1];
    if check_convex {
        for row in weights.data.chunks_exact(k) {
            ensure(
                row.iter().all(|&w| w >= -1e-6) && (row.iter().sum::<f64>() - 1.).abs() <= 1e-6,
                "resampling needs convex weights summing to one",
            )?;
        }
    }
    let tv = d.get("template_vertices")?;
    let shapes = d.get("blendshapes")?;
    let c = d.blendshape_count();
    let mut new_vertices = Tensor::zeros(vec![n, 3]);
    let mut new_shapes = Tensor::zeros(vec![c, n, 3]);
    for v in 0..n {
        for s in 0..k {
            let id = ids[v * k + s];
            let w = weights.data[v * k + s];
            for a in 0..3 {
                new_vertices.data[v * 3 + a] += w * tv.data[id * 3 + a];
            }
            if w != 0. {
                for b in 0..c {
                    for a in 0..3 {
                        new_shapes.data[(b * n + v) * 3 + a] +=
                            w * shapes.data[(b * old_n + id) * 3 + a];
                    }
                }
            }
        }
    }
    if let Some(v) = vertices {
        v.expect_shape(&[n, 3], "target vertices")?;
        new_vertices = v;
    }
    faces.checked_indices(n, "resampled faces")?;
    let old_w = d.get("vertex_bone_weights")?;
    let old_i = d.get("vertex_bone_indices")?;
    let slots = old_w.shape[1];
    let bones = d.bone_count();
    let mut entries = Vec::new();
    let mut width = 1;
    for v in 0..n {
        let mut total = vec![0.; bones];
        let mut seen = vec![false; bones];
        let mut order = Vec::new();
        for s in 0..k {
            let id = ids[v * k + s];
            for slot in 0..slots {
                let bone = old_i.data[id * slots + slot] as usize;
                if !seen[bone] {
                    seen[bone] = true;
                    order.push(bone);
                }
                total[bone] += weights.data[v * k + s] * old_w.data[id * slots + slot];
            }
        }
        let mut row: Vec<_> = order
            .into_iter()
            .filter(|&b| total[b].abs() > 1e-12)
            .map(|b| (b, total[b]))
            .collect();
        let sum: f64 = row.iter().map(|x| x.1).sum();
        ensure(
            sum.abs() >= 1e-12,
            format!("resampling left vertex {v} unbound"),
        )?;
        for (_, w) in &mut row {
            *w /= sum;
        }
        width = width.max(row.len());
        entries.push(row);
    }
    let mut ws = Tensor::zeros(vec![n, width]);
    let mut is = ws.clone();
    is.kind = Kind::Index;
    for (v, row) in entries.iter().enumerate() {
        for (s, &(b, w)) in row.iter().enumerate() {
            ws.data[v * width + s] = w;
            is.data[v * width + s] = b as f64;
        }
    }
    let base = base_indices.unwrap_or_else(|| Tensor::indices(vec![n], (0..n).collect()));
    base.expect_shape(&[n], "base indices")?;
    d.put("template_vertices", new_vertices);
    d.put("blendshapes", new_shapes);
    d.put("faces", faces);
    d.put("vertex_bone_weights", ws);
    d.put("vertex_bone_indices", is);
    d.put("base_mesh_vertex_indices", base);
    d.remove(&["texture_coordinates", "face_texture_coordinate_indices"]);
    Ok(())
}
fn triangulated(vertices: &Tensor, faces: &Tensor) -> Result<Tensor> {
    let mut temp = ModelData::default();
    temp.put("template_vertices", vertices.clone());
    temp.put("faces", faces.clone());
    mesh::triangulate(&mut temp)?;
    Ok(temp.arrays.remove("faces").unwrap())
}
pub fn projection_coordinates(
    target: &Tensor,
    source: &Tensor,
    faces: &Tensor,
    max_distance: Option<f64>,
) -> Result<(Tensor, Tensor)> {
    let triangles = triangulated(source, faces)?;
    let mut rounded = source.clone();
    for v in &mut rounded.data {
        *v = (*v as f32) as f64;
    }
    let bvh = mesh::MeshBvh::new_projection(&rounded, &triangles)?;
    let mut indices = Vec::new();
    let mut weights = Vec::new();
    for p in target.data.chunks_exact(3) {
        let p = Vec3::new(p[0] as f32 as f64, p[1] as f32 as f64, p[2] as f32 as f64);
        let (dist, id, w) = bvh.closest_portable_f32(p);
        ensure(
            id != usize::MAX,
            "projection found no nondegenerate triangle",
        )?;
        if let Some(max) = max_distance {
            ensure(
                dist < max,
                format!("target is {dist} m from reference mesh; maximum {max}"),
            )?;
        }
        indices.extend(bvh.triangle_indices(id));
        let u = w[0] as f32 as f64;
        let v = w[1] as f32 as f64;
        weights.extend([u, v, 1. - u - v]);
    }
    Ok((
        Tensor::indices(vec![target.shape[0], 3], indices),
        Tensor::new(vec![target.shape[0], 3], weights)?,
    ))
}
pub fn apply_retopology_from_mesh(
    d: &mut ModelData,
    target: &Tensor,
    faces: Tensor,
    source: &Tensor,
    source_faces: &Tensor,
) -> Result<()> {
    ensure(
        source.shape == [d.vertex_count(), 3],
        "projection reference must share model vertex ordering",
    )?;
    let (ids, weights) = projection_coordinates(target, source, source_faces, Some(0.015))?;
    interpolate_model_data(
        d,
        &ids,
        &weights,
        triangulated(target, &faces)?,
        None,
        None,
        true,
    )
}
fn payload_barycentric(a: &Archive, key: &str, n: usize, k: usize) -> Result<Tensor> {
    if let Ok(t) = a.payload_tensor(key) {
        if t.shape == [n, k] {
            return Ok(t.clone());
        }
        if t.shape == [k, n] {
            let mut out = Tensor::zeros(vec![n, k]);
            for i in 0..n {
                for s in 0..k {
                    out.data[i * k + s] = t.data[s * n + i];
                }
            }
            return Ok(out);
        }
    }
    let p = a.payload()?;
    let list = p[key]
        .as_array()
        .ok_or_else(|| Error::Invalid("invalid barycentric payload".into()))?;
    ensure(list.len() == k, "invalid barycentric columns")?;
    let mut out = Tensor::zeros(vec![n, k]);
    for (s, col) in list.iter().enumerate() {
        let name = col["__tensor__"]
            .as_str()
            .ok_or_else(|| Error::Invalid("missing barycentric tensor".into()))?;
        let t = a.tensor(name)?;
        t.expect_shape(&[n], "barycentric column")?;
        for i in 0..n {
            out.data[i * k + s] = t.data[i];
        }
    }
    Ok(out)
}
pub fn apply_soma_rig(d: &mut ModelData, a: &Archive, cov: &Archive) -> Result<()> {
    let payload = a.payload()?;
    let cp = cov.payload()?;
    let labels: Vec<String> = serde_json::from_value(payload["bone_labels"].clone())?;
    let parents: Vec<i32> = serde_json::from_value(payload["bone_parents"].clone())?;
    ensure(
        payload["bone_labels"] == cp["bone_labels"],
        "SOMA rig and cache bone labels disagree",
    )?;
    let source_shapes: Vec<String> = serde_json::from_value(cp["blendshape_labels"].clone())?;
    let n = d.vertex_count();
    let j = labels.len();
    let c = d.blendshape_count();
    let rbf = a.payload_tensor("sparse_rbf_matrix")?;
    rbf.expect_shape(&[j, n], "SOMA origin regression")?;
    let tv = d.get("template_vertices")?;
    let bs = d.get("blendshapes")?;
    let mut heads = Tensor::zeros(vec![j, 3]);
    let mut head_shapes = Tensor::zeros(vec![c, j, 3]);
    for bone in 0..j {
        for v in 0..n {
            let w = rbf.data[bone * n + v];
            if w == 0. {
                continue;
            }
            for axis in 0..3 {
                heads.data[bone * 3 + axis] += w * tv.data[v * 3 + axis];
            }
            for b in 0..c {
                for axis in 0..3 {
                    head_shapes.data[(b * j + bone) * 3 + axis] +=
                        w * bs.data[(b * n + v) * 3 + axis];
                }
            }
        }
    }
    for axis in 0..3 {
        heads.data[axis] = heads.data[3 + axis];
    }
    for b in 0..c {
        for axis in 0..3 {
            head_shapes.data[b * j * 3 + axis] = head_shapes.data[(b * j + 1) * 3 + axis];
        }
    }
    let raw = a.payload_tensor("skinning_weights")?;
    raw.expect_shape(&[n, j], "SOMA skinning")?;
    let k = raw
        .data
        .chunks_exact(j)
        .map(|row| row.iter().filter(|&&w| w > 0.).count())
        .max()
        .unwrap_or(1)
        .max(1);
    let mut weights = Tensor::zeros(vec![n, k]);
    let mut indices = weights.clone();
    indices.kind = Kind::Index;
    for (v, row) in raw.data.chunks_exact(j).enumerate() {
        let mut order: Vec<_> = (0..j).collect();
        order.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
        for s in 0..k {
            indices.data[v * k + s] = order[s] as f64;
            weights.data[v * k + s] = row[order[s]];
        }
    }
    let bind = a.payload_tensor("bind_world_transforms")?;
    bind.expect_shape(&[j, 4, 4], "SOMA bind poses")?;
    let bind: Vec<_> = bind.data.chunks_exact(16).map(mat4).collect();
    let tpose = a.payload_tensor("t_pose_world")?;
    tpose.expect_shape(&[j, 4, 4], "SOMA t-pose")?;
    let mut reference = Tensor::zeros(vec![j, 3, 3]);
    for (i, t) in tpose.data.chunks_exact(16).enumerate() {
        write3(&rotation(&mat4(t)), &mut reference.data[i * 9..(i + 1) * 9]);
    }
    let mut children = vec![Vec::new(); j];
    for (i, &p) in parents.iter().enumerate() {
        if p >= 0 {
            children[p as usize].push(i);
        }
    }
    let max = children.iter().map(Vec::len).max().unwrap_or(0);
    let mut child_i = Tensor::zeros(vec![j, max]);
    child_i.kind = Kind::Index;
    let mut child_m = Tensor::zeros(vec![j, max]);
    let mut offsets = Tensor::zeros(vec![j, max, 3]);
    for bone in 0..j {
        for (s, &child) in children[bone].iter().enumerate() {
            child_i.data[bone * max + s] = child as f64;
            child_m.data[bone * max + s] = 1.;
            let off = rotation(&bind[bone]).transpose()
                * (translation(&bind[child]) - translation(&bind[bone]));
            offsets.data[(bone * max + s) * 3..(bone * max + s + 1) * 3]
                .copy_from_slice(off.as_slice());
        }
    }
    let orientation = select_blendshape_rows(
        cov.payload_tensor("bone_orientation_blendshapes")?,
        &source_shapes,
        &d.metadata.blendshape_labels,
    )?;
    d.metadata.bone_labels = labels;
    d.metadata.bone_parents = parents;
    for (name, t) in [
        ("template_bone_heads", heads),
        ("bone_heads_blendshapes", head_shapes),
        ("vertex_bone_weights", weights),
        ("vertex_bone_indices", indices),
        (
            "bone_template_orientation_matrices",
            cov.payload_tensor("bone_template_orientation_matrices")?
                .clone(),
        ),
        ("bone_orientation_blendshapes", orientation),
        ("reference_bone_orientations", reference),
        ("bone_children_indices", child_i),
        ("bone_children_mask", child_m),
        ("bone_children_local_offsets", offsets),
    ] {
        d.put(name, t);
    }
    d.remove(&[
        "template_bone_tails",
        "bone_tails_blendshapes",
        "bone_rolls_rotmat",
        "bone_nonzeroweight_mask",
        "bone_vertex_indices",
        "bone_vertex_weights",
        "template_bone_vertices",
    ]);
    Ok(())
}
