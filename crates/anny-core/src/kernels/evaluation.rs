// Shared f32/f64 model evaluation; scalar arithmetic is selected at compile time.
pub fn broadcast(sizes: &[usize]) -> Result<usize> {
    let b = *sizes.iter().max().unwrap_or(&1);
    ensure(
        b > 0 && sizes.iter().all(|&n| n == 1 || n == b),
        format!("incompatible batch sizes {sizes:?}"),
    )?;
    Ok(b)
}
pub fn parse_values(
    value: &Value,
    labels: &[String],
    default: Scalar,
    name: &str,
) -> Result<Tensor> {
    if value.is_null() {
        return Tensor::new(vec![1, labels.len()], vec![default; labels.len()]);
    }
    if let Some(map) = value.as_object() {
        for key in map.keys() {
            ensure(
                labels.contains(key),
                format!("unknown {name} label {key}; available: {labels:?}"),
            )?;
        }
        let rows: Vec<Vec<Scalar>> = labels
            .iter()
            .map(|key| match map.get(key) {
                None => Ok(vec![default]),
                Some(v) if v.is_number() => Ok(vec![v.as_f64().unwrap() as Scalar]),
                Some(v) => {
                    let t = Tensor::from_nested(v)?;
                    ensure(
                        t.shape.len() == 1,
                        format!("{name} values must be scalar or 1-D"),
                    )?;
                    Ok(t.data)
                }
            })
            .collect::<Result<_>>()?;
        let b = broadcast(&rows.iter().map(Vec::len).collect::<Vec<_>>())?;
        let mut data = Vec::with_capacity(b * labels.len());
        for i in 0..b {
            for r in &rows {
                data.push(r[i % r.len()]);
            }
        }
        Tensor::new(vec![b, labels.len()], data)
    } else {
        let t = Tensor::from_nested(value)?;
        ensure(
            t.shape.len() == 2 && t.shape[0] > 0 && t.shape[1] == labels.len(),
            format!("{name} must have shape [B,{}]", labels.len()),
        )?;
        Ok(t)
    }
}
pub fn parse_pose(value: &Value, labels: &[String]) -> Result<Tensor> {
    let j = labels.len();
    if let Some(map) = value.as_object() {
        for key in map.keys() {
            ensure(labels.contains(key), format!("unknown bone {key}"))?;
        }
        let mut entries = Vec::new();
        let mut sizes = vec![1];
        for (key, v) in map {
            let mut t = Tensor::from_nested(v)?;
            if t.shape == [4, 4] {
                t.shape.insert(0, 1);
            }
            ensure(
                t.shape.len() == 3 && t.shape[1..] == [4, 4],
                "named poses must be [4,4] or [B,4,4]",
            )?;
            sizes.push(t.shape[0]);
            entries.push((labels.iter().position(|x| x == key).unwrap(), t));
        }
        let b = broadcast(&sizes)?;
        let mut out = identity_poses(b, j);
        for (i, t) in entries {
            for k in 0..b {
                out.data[(k * j + i) * 16..(k * j + i + 1) * 16]
                    .copy_from_slice(&t.data[(k % t.shape[0]) * 16..(k % t.shape[0] + 1) * 16]);
            }
        }
        validate_pose(&out)?;
        Ok(out)
    } else if value.is_null() {
        Ok(identity_poses(1, j))
    } else {
        let mut t = Tensor::from_nested(value)?;
        if t.shape == [j, 4, 4] {
            t.shape.insert(0, 1);
        }
        ensure(
            t.shape.len() == 4 && t.shape[0] > 0 && t.shape[1..] == [j, 4, 4],
            format!("pose must have shape [B,{j},4,4]"),
        )?;
        validate_pose(&t)?;
        Ok(t)
    }
}
pub fn identity_poses(b: usize, j: usize) -> Tensor {
    let mut t = Tensor::zeros(vec![b, j, 4, 4]);
    for row in t.data.chunks_exact_mut(16) {
        for i in 0..4 {
            row[i * 4 + i] = 1.;
        }
    }
    t
}
fn validate_pose(t: &Tensor) -> Result<()> {
    for m in t.data.chunks_exact(16) {
        checked_rigid(&mat4(m))?;
    }
    Ok(())
}
fn validate_orientation(d: &ModelData, orientation: BoneOrientation) -> Result<()> {
    let fields: &[&str] = match orientation {
        BoneOrientation::Blender => &[
            "template_bone_tails",
            "bone_tails_blendshapes",
            "bone_rolls_rotmat",
        ],
        BoneOrientation::Cached => &[
            "bone_template_orientation_matrices",
            "bone_orientation_blendshapes",
        ],
        BoneOrientation::Procrustes => &[
            "bone_nonzeroweight_mask",
            "bone_vertex_indices",
            "bone_vertex_weights",
            "template_bone_vertices",
        ],
    };
    for f in fields {
        d.get(f)?;
    }
    if orientation == BoneOrientation::Procrustes {
        let mask = d.get("bone_nonzeroweight_mask")?;
        mask.expect_shape(&[d.bone_count()], "bone mask")?;
        let a = mask.data.iter().filter(|&&x| x != 0.).count();
        let idx = d.get("bone_vertex_indices")?;
        ensure(
            idx.shape.len() == 2 && idx.shape[0] == a,
            "invalid procrustes sample shape",
        )?;
        idx.checked_indices(d.vertex_count(), "procrustes indices")?;
        d.get("bone_vertex_weights")?
            .expect_shape(&idx.shape, "procrustes weights")?;
        d.get("template_bone_vertices")?
            .expect_shape(&[a, idx.shape[1], 3], "procrustes vertices")?;
    }
    if let Some(idx) = d.arrays.get("bone_children_indices") {
        ensure(
            idx.shape.len() == 2 && idx.shape[0] == d.bone_count(),
            "invalid child indices shape",
        )?;
        idx.checked_indices(d.bone_count(), "child indices")?;
        d.get("bone_children_mask")?
            .expect_shape(&idx.shape, "child mask")?;
        d.get("bone_children_local_offsets")?
            .expect_shape(&[d.bone_count(), idx.shape[1], 3], "child offsets")?;
    }
    Ok(())
}
/// Applies C x N x D blendshapes to a N x D template, yielding B x N x D.
pub fn apply_blendshapes(
    template: &Tensor,
    blendshapes: &Tensor,
    coeffs: &Tensor,
) -> Result<Tensor> {
    ensure(
        !template.shape.is_empty() && coeffs.shape.len() == 2,
        "invalid blendshape tensor ranks",
    )?;
    let c = coeffs.shape[1];
    let size = template.data.len();
    let mut expected = vec![c];
    expected.extend_from_slice(&template.shape);
    blendshapes.expect_shape(&expected, "blendshapes")?;
    let mut shape = vec![coeffs.shape[0]];
    shape.extend_from_slice(&template.shape);
    let mut out = Tensor::zeros(shape);
    for (b, row) in out.data.chunks_exact_mut(size).enumerate() {
        row.copy_from_slice(&template.data);
        for k in 0..c {
            let w = coeffs.data[b * c + k];
            if w == 0. {
                continue;
            }
            for (v, delta) in row
                .iter_mut()
                .zip(&blendshapes.data[k * size..(k + 1) * size])
            {
                *v += w * delta;
            }
        }
    }
    Ok(out)
}

pub fn rest_model(d: &ModelData, rig: &RigConfig, coeffs: &Tensor) -> Result<ModelOutput> {
    ensure(
        coeffs.shape.len() == 2 && coeffs.shape[0] > 0 && coeffs.shape[1] == d.blendshape_count(),
        "blendshape coefficients must be [B,C]",
    )?;
    coeffs.validate()?;
    let rest = apply_blendshapes(d.get("template_vertices")?, d.get("blendshapes")?, coeffs)?;
    let heads = apply_blendshapes(
        d.get("template_bone_heads")?,
        d.get("bone_heads_blendshapes")?,
        coeffs,
    )?;
    let b = coeffs.shape[0];
    let j = d.bone_count();
    let n = d.vertex_count();
    let mut poses = identity_poses(b, j);
    let mut tails = None;
    let mut orientations = vec![Mat3::identity(); b * j];
    match rig.bone_orientation {
        BoneOrientation::Blender => {
            let ts = apply_blendshapes(
                d.get("template_bone_tails")?,
                d.get("bone_tails_blendshapes")?,
                coeffs,
            )?;
            let rolls = d.get("bone_rolls_rotmat")?;
            for (i, orientation) in orientations.iter_mut().enumerate() {
                *orientation = rotation(&tail_pose(
                    &vec3(&heads.data[i * 3..i * 3 + 3]),
                    &vec3(&ts.data[i * 3..i * 3 + 3]),
                    &mat3(&rolls.data[(i % j) * 9..(i % j + 1) * 9]),
                ));
            }
            tails = Some(ts);
        }
        BoneOrientation::Cached => {
            let cov = apply_blendshapes(
                d.get("bone_template_orientation_matrices")?,
                d.get("bone_orientation_blendshapes")?,
                coeffs,
            )?;
            for (o, m) in orientations.iter_mut().zip(cov.data.chunks_exact(9)) {
                *o = special_procrustes(&mat3(m));
            }
            if d.arrays.contains_key("bone_children_indices") {
                refine_orientations(d, &heads, &mut orientations)?;
            }
        }
        BoneOrientation::Procrustes => {
            let ids = d.get("bone_vertex_indices")?;
            let weights = d.get("bone_vertex_weights")?;
            let source = d.get("template_bone_vertices")?;
            let active: Vec<_> = d
                .get("bone_nonzeroweight_mask")?
                .data
                .iter()
                .enumerate()
                .filter(|(_, w)| **w != 0.)
                .map(|(i, _)| i)
                .collect();
            let k = ids.shape[1];
            for bi in 0..b {
                for (row, &bone) in active.iter().enumerate() {
                    let mut h = Mat3::zeros();
                    let head = vec3(&heads.data[(bi * j + bone) * 3..(bi * j + bone + 1) * 3]);
                    for q in 0..k {
                        let id = ids.data[row * k + q] as usize;
                        let w = weights.data[row * k + q];
                        if w == 0. {
                            continue;
                        }
                        let v = vec3(&rest.data[(bi * n + id) * 3..(bi * n + id + 1) * 3]) - head;
                        let s = vec3(&source.data[(row * k + q) * 3..(row * k + q + 1) * 3]);
                        h += w * v * s.transpose();
                    }
                    orientations[bi * j + bone] = special_procrustes(&h);
                }
            }
        }
    }
    for (i, orientation) in orientations.iter_mut().enumerate() {
        if i % j == 0 && rig.root_identity_orientation {
            *orientation = Mat3::identity();
        }
        write4(
            &rigid(orientation, &vec3(&heads.data[i * 3..i * 3 + 3])),
            &mut poses.data[i * 16..(i + 1) * 16],
        );
    }
    let mut arrays = BTreeMap::from([
        ("rest_vertices".into(), rest),
        ("rest_bone_heads".into(), heads),
        ("rest_bone_poses".into(), poses),
    ]);
    if let Some(t) = tails {
        arrays.insert("rest_bone_tails".into(), t);
    }
    Ok(ModelOutput { arrays })
}
fn refine_orientations(d: &ModelData, heads: &Tensor, orientations: &mut [Mat3]) -> Result<()> {
    let idx = d.get("bone_children_indices")?;
    let mask = d.get("bone_children_mask")?;
    let off = d.get("bone_children_local_offsets")?;
    let j = d.bone_count();
    let k = idx.shape[1];
    let original = orientations.to_vec();
    for bi in 0..heads.shape[0] {
        for bone in 1..j {
            let children: Vec<_> = (0..k).filter(|&i| mask.data[bone * k + i] != 0.).collect();
            if children.is_empty() {
                continue;
            }
            let center = vec3(&heads.data[(bi * j + bone) * 3..(bi * j + bone + 1) * 3]);
            let mut a = Vec::new();
            let mut b = Vec::new();
            for &i in &children {
                let id = idx.data[bone * k + i] as usize;
                a.push(vec3(&heads.data[(bi * j + id) * 3..(bi * j + id + 1) * 3]) - center);
                b.push(
                    original[bi * j + bone]
                        * vec3(&off.data[(bone * k + i) * 3..(bone * k + i + 1) * 3]),
                );
            }
            let align = if a.len() == 1 {
                shortest_arc(&a[0], &b[0])
            } else {
                let mut h = Mat3::zeros();
                for (a, b) in a.iter().zip(&b) {
                    h += a * b.transpose();
                }
                let na = a[0].cross(&a[1]);
                let nb = b[0].cross(&b[1]);
                if na.norm() > 1e-9 && nb.norm() > 1e-9 {
                    let va = na * a[0].norm() / (na.norm() + 1e-8);
                    let vb = nb * b[0].norm() / (nb.norm() + 1e-8);
                    h += va * vb.transpose();
                }
                special_procrustes(&h)
            };
            orientations[bi * j + bone] = align * original[bi * j + bone];
        }
        let before = orientations[bi * j..(bi + 1) * j].to_vec();
        for bone in 1..j {
            if !d.metadata.bone_parents.contains(&(bone as i32)) {
                orientations[bi * j + bone] = before[d.metadata.bone_parents[bone] as usize];
            }
        }
    }
    Ok(())
}
fn reference_poses(d: &ModelData, rest: &[Mat4]) -> Result<Vec<Mat4>> {
    if let Some(r) = d.arrays.get("reference_bone_orientations") {
        let rs: Vec<_> = r.data.chunks_exact(9).map(mat3).collect();
        Ok(absolute_kinematics(&d.metadata.bone_parents, rest, &rs, None)?.0)
    } else {
        Ok(rest.to_vec())
    }
}
pub fn bone_transforms(
    d: &ModelData,
    rest: &[Mat4],
    delta: &[Mat4],
    mode: PoseParameterization,
) -> Result<(Vec<Mat4>, Vec<Mat4>)> {
    let (poses, transforms) = match mode {
        PoseParameterization::World => (delta.to_vec(), None),
        PoseParameterization::WorldOrient => {
            let base = delta[0] * inverse_rigid(&rest[0]);
            let r: Vec<_> = delta.iter().map(rotation).collect();
            let (p, t) = absolute_kinematics(&d.metadata.bone_parents, rest, &r, Some(&base))?;
            (p, Some(t))
        }
        _ => {
            let refs = reference_poses(d, rest)?;
            let base = if mode == PoseParameterization::LocalBoneWorld {
                None
            } else {
                Some(inverse_rigid(&refs[0]))
            };
            let deltas: Vec<_> = delta
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    if mode == PoseParameterization::LocalRef {
                        let r = rigid(&rotation(&refs[i]), &Vec3::zeros());
                        inverse_rigid(&r) * t * r
                    } else {
                        *t
                    }
                })
                .collect();
            let (p, _) =
                forward_kinematics(&d.metadata.bone_parents, &refs, &deltas, base.as_ref())?;
            (p, None)
        }
    };
    let transforms = transforms.unwrap_or_else(|| {
        poses
            .iter()
            .zip(rest)
            .map(|(p, r)| p * inverse_rigid(r))
            .collect()
    });
    Ok((transforms, poses))
}
pub fn forward_model(
    d: &ModelData,
    rig: &RigConfig,
    coeffs: &Tensor,
    pose: &Value,
    mode: PoseParameterization,
    skinning: SkinningMethod,
    return_bone_ends: bool,
) -> Result<ModelOutput> {
    let out = rest_model(d, rig, coeffs)?;
    pose_model(d, rig, out, pose, mode, skinning, return_bone_ends)
}
/// Take an output buffer for reuse when its shape already matches, otherwise allocate.
///
/// A pose-only update reproduces the same shapes call after call, so stealing the previous buffer
/// keeps the animating path allocation-free. The buffer is zeroed because callers rely on
/// `Tensor::zeros` semantics.
fn reusable_buffer(out: &mut ModelOutput, key: &str, shape: Vec<usize>) -> Tensor {
    match out.arrays.remove(key) {
        Some(mut existing) if existing.shape == shape => {
            existing.data.fill(0.);
            existing
        }
        _ => Tensor::zeros(shape),
    }
}
/// Reuse an output buffer for bone poses, re-initialised to identity.
///
/// The pose loop overwrites the matrices it has poses for; starting from identity (as
/// `identity_poses` does) keeps the result identical whether the buffer was reused or new.
fn reusable_identity(out: &mut ModelOutput, key: &str, b: usize, j: usize) -> Tensor {
    // Bone poses are [b, j, 4, 4], matching `identity_poses`; the rank is load-bearing because
    // consumers such as `scene::selected` validate the trailing dimensions.
    let mut t = reusable_buffer(out, key, vec![b, j, 4, 4]);
    let identity = identity_poses(1, j);
    for bi in 0..b {
        t.data[bi * j * 16..(bi + 1) * j * 16].copy_from_slice(&identity.data);
    }
    t
}
/// Evaluate a pose against an already-built rest model.
///
/// This is the pose-dependent half of [`forward_model`]. It is separate so that a caller who keeps
/// the phenotype/local-change coefficients fixed (animation, re-posing, editor sliders) can reuse
/// the rest model instead of rebuilding it on every update; see `Anny::pose_session`.
///
/// `out` must be the output of [`rest_model`] for the same coefficients, and it is consumed so the
/// rest arrays it already holds are carried into the returned output rather than recomputed.
pub fn pose_model(
    d: &ModelData,
    rig: &RigConfig,
    mut out: ModelOutput,
    pose: &Value,
    mode: PoseParameterization,
    skinning: SkinningMethod,
    return_bone_ends: bool,
) -> Result<ModelOutput> {
    let deltas = parse_pose(pose, &d.metadata.bone_labels)?;
    let br = out.get("rest_bone_poses")?.shape[0];
    let b = broadcast(&[br, deltas.shape[0]])?;
    let j = d.bone_count();
    let n = d.vertex_count();
    // Acquire every output buffer before the rest arrays are borrowed, so the borrow checker stays
    // happy without cloning and a repeated update reuses its previous buffers.
    let mut posed = reusable_buffer(&mut out, "vertices", vec![b, n, 3]);
    let mut bones = reusable_identity(&mut out, "bone_poses", b, j);
    let ends_buffers = if return_bone_ends {
        Some((
            reusable_buffer(&mut out, "bone_heads", vec![b, j, 3]),
            reusable_buffer(&mut out, "bone_tails", vec![b, j, 3]),
        ))
    } else {
        out.arrays.remove("bone_heads");
        out.arrays.remove("bone_tails");
        None
    };
    let weights = d.get("vertex_bone_weights")?;
    let ids = d.get("vertex_bone_indices")?;
    let k = weights.shape[1];
    let rest = out.get("rest_vertices")?;
    let restposes = out.get("rest_bone_poses")?;
    let mut ends = if let Some(buffers) = ends_buffers {
        ensure(
            rig.bone_orientation == BoneOrientation::Blender,
            "bone ends require blender/tail orientation",
        )?;
        Some(buffers)
    } else {
        None
    };
    for bi in 0..b {
        let rp: Vec<_> = restposes.data[(bi % br) * j * 16..(bi % br + 1) * j * 16]
            .chunks_exact(16)
            .map(mat4)
            .collect();
        let delta: Vec<_> = deltas.data
            [(bi % deltas.shape[0]) * j * 16..(bi % deltas.shape[0] + 1) * j * 16]
            .chunks_exact(16)
            .map(mat4)
            .collect();
        let (transforms, poses) = bone_transforms(d, &rp, &delta, mode)?;
        let dqs: Vec<_> = if skinning == SkinningMethod::Dqs {
            transforms.iter().map(dual_quaternion).collect()
        } else {
            vec![]
        };
        for v in 0..n {
            let xyz = vec3(&rest.data[((bi % br) * n + v) * 3..((bi % br) * n + v + 1) * 3]);
            let p = if skinning == SkinningMethod::Dqs {
                dqs_point(
                    &xyz,
                    (0..k).map(|s| {
                        let index = v * k + s;
                        (weights.data[index], dqs[ids.data[index] as usize])
                    }),
                )?
            } else {
                let mut p = Vec3::zeros();
                for s in 0..k {
                    let index = v * k + s;
                    let w = weights.data[index];
                    if w != 0. {
                        p += w * point(&transforms[ids.data[index] as usize], &xyz);
                    }
                }
                p
            };
            posed.data[(bi * n + v) * 3..(bi * n + v + 1) * 3].copy_from_slice(p.as_slice());
        }
        for (i, p) in poses.iter().enumerate() {
            write4(p, &mut bones.data[(bi * j + i) * 16..(bi * j + i + 1) * 16]);
        }
        if let Some((heads, tails)) = &mut ends {
            let rh = out.get("rest_bone_heads")?;
            let rt = out.get("rest_bone_tails")?;
            for (i, transform) in transforms.iter().enumerate() {
                let h = point(
                    transform,
                    &vec3(&rh.data[((bi % br) * j + i) * 3..((bi % br) * j + i + 1) * 3]),
                );
                let t = point(
                    transform,
                    &vec3(&rt.data[((bi % br) * j + i) * 3..((bi % br) * j + i + 1) * 3]),
                );
                heads.data[(bi * j + i) * 3..(bi * j + i + 1) * 3].copy_from_slice(h.as_slice());
                tails.data[(bi * j + i) * 3..(bi * j + i + 1) * 3].copy_from_slice(t.as_slice());
            }
        }
    }
    posed.validate()?;
    bones.validate()?;
    out.arrays.insert("vertices".into(), posed);
    out.arrays.insert("bone_poses".into(), bones);
    if let Some((h, t)) = ends {
        out.arrays.insert("bone_heads".into(), h);
        out.arrays.insert("bone_tails".into(), t);
    }
    Ok(out)
}
pub fn pose_parameters(
    d: &ModelData,
    out: &ModelOutput,
    mode: PoseParameterization,
) -> Result<Tensor> {
    let bp = out.get("bone_poses")?;
    let rest = out.get("rest_bone_poses")?;
    let b = bp.shape[0];
    let j = d.bone_count();
    let br = rest.shape[0];
    let mut result = identity_poses(b, j);
    for bi in 0..b {
        let poses: Vec<_> = bp.data[bi * j * 16..(bi + 1) * j * 16]
            .chunks_exact(16)
            .map(mat4)
            .collect();
        let rp: Vec<_> = rest.data[(bi % br) * j * 16..(bi % br + 1) * j * 16]
            .chunks_exact(16)
            .map(mat4)
            .collect();
        let refs = reference_poses(d, &rp)?;
        for i in 0..j {
            let mut p = match mode {
                PoseParameterization::World => poses[i],
                PoseParameterization::WorldOrient => {
                    if i == 0 {
                        poses[i]
                    } else {
                        rigid(&rotation(&poses[i]), &Vec3::zeros())
                    }
                }
                _ if i == 0 => {
                    if mode == PoseParameterization::LocalBoneWorld {
                        inverse_rigid(&refs[0]) * poses[0]
                    } else {
                        poses[0]
                    }
                }
                _ => {
                    let parent = d.metadata.bone_parents[i] as usize;
                    let relative_ref = inverse_rigid(&refs[parent]) * refs[i];
                    let relative = inverse_rigid(&poses[parent]) * poses[i];
                    inverse(&relative_ref)? * relative
                }
            };
            if mode == PoseParameterization::LocalRef {
                let r = rigid(&rotation(&refs[i]), &Vec3::zeros());
                p = r * p * inverse_rigid(&r);
            }
            write4(
                &p,
                &mut result.data[(bi * j + i) * 16..(bi * j + i + 1) * 16],
            );
        }
    }
    Ok(result)
}
