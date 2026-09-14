# Upstream parity audit

Inventory of usable capabilities in the pinned upstream NAVER Anny checkout against the Rust
workspace.

| | |
|---|---|
| Upstream revision | `naver/anny@81ca83e202273b306205c1cc15f33734be31e48c` |
| Upstream checkout used | `/home/slurp/Source/Not_Saved/anny` |
| ModelData schema | 11 |
| Audit tool | `tools/upstream_audit.py` |

`tools/upstream_audit.py` extracts every public module-level function and public class method from
the upstream Python packages and matches names against the Rust workspace. Name matches alone are
not the result of the audit: every miss was reviewed by hand for a semantic counterpart that simply
has a different name in Rust (the common case, because the port is written as native kernels rather
than as a transcription of the Python API). Only capabilities with no counterpart at all are listed
in [Genuinely absent](#genuinely-absent).

This document is a mapping, not a claim of bit-for-bit behavioural identity. Where the port
deliberately differs it says so and says why.

## Upstream examples and scripts

| Upstream | Native equivalent | State |
|---|---|---|
| `examples/amass_to_anny.py` | `anny amass-inspect`, `anny amass-fit`; `anny_core::motion` | DONE |
| `examples/benchmark.py` | `anny benchmark` (single-configuration forward timing with a JSON report) and `cargo bench -p anny-core` (configuration matrix, batching, prepare/reload, secondary operations) | DONE — see `docs/PERFORMANCE.md` |
| `examples/benchmark_compile.py` | Not applicable as such: the native build is already ahead-of-time compiled. The useful goal (measure the compiled path against a naive one) is covered by the same native benchmarks. | NOT REQUIRED (literal `torch.compile`) |
| `examples/mesh_to_params.py` | `anny fit`, `anny_core::fitting` (known-index, closest-surface, landmark similarity init, multistart) | DONE |
| `examples/smpl_comparison.py` | `anny_core::smpl` SMPL/SMPL-X adapter; caller-supplied licensed model | DONE (capability) / EXTERNAL (data) |
| `examples/smplx_comparison.py` | Same adapter, SMPL-X path | DONE (capability) / EXTERNAL (data) |
| `examples/soma_comparison.py` | `RigSpec::Soma`, `precompute_soma`, SOMA parity case | DONE |
| `scripts/compute_skinning_weights.py` | `anny_core::precompute` (`precompute_skinning_weights`) | DONE |
| `scripts/generate_regression_fixtures.py` | `tools/export_reference.py` (reference oracle side only) | DONE (developer tool) |
| `scripts/precompute_rig_caches.py` | `anny_core::precompute::precompute_anny`, `anny prepare` | DONE |
| `scripts/precompute_soma_rig.py` | `anny_core::precompute::precompute_soma` | DONE |

## `model_transforms.py`

The whole surface is present in `crates/anny-core/src/transforms.rs`. `posterior_covariance` exists
under the same name.

| Upstream | Rust |
|---|---|
| `filter_blendshapes` | `transforms::filter_blendshapes` |
| `filter_faces` | `transforms::filter_faces` |
| `filter_vertices` | `transforms::filter_vertices` |
| `filter_position_independent_vertex_bone_weights` | `transforms::filter_position_independent_vertex_bone_weights` |
| `triangulate` | `transforms::triangulate` |
| `edit_mesh` | `transforms::edit_mesh` |
| `remove_unattached_vertices` | `transforms::remove_unattached_vertices` |
| `symmetrize_skinning_weights` | `transforms::symmetrize_skinning_weights` |
| `remove_skinning_islands` | `transforms::remove_skinning_islands` |
| `compact_skinning_weights` | `transforms::compact_skinning_weights` |
| `apply_procrustes_orientation` | `transforms::apply_procrustes_orientation` |
| `apply_cached_orientation` | `assets::AssetStore::cached_orientation` |
| `apply_rest_bone_orientations` | `transforms::` rest-orientation helpers |
| `with_bone_orientation` | Rig construction in `assets.rs` (`BoneOrientation::{Blender,Procrustes,Cached}`) |
| `with_bone_rest_pose` | Rig construction in `assets.rs` |
| `split_skinning_weights`, `split_vertex_bone_weights` | Compact/expand weight handling in `transforms.rs` |
| `posterior_covariance` | `transforms::posterior_covariance` |
| `rig transforms` (reparenting, aggregation, filtering) | `assets.rs` rig selection + `transforms.rs` |

Covered by `crates/anny-core/tests/authoring.rs`.

## Upstream utilities

| Upstream module | Native equivalent | State |
|---|---|---|
| `utils/collision.py` — `detect_self_intersections`, `SelfInterpenetrationModule`, the Warp intersection kernel | `crates/anny-core/src/tools.rs` self-interpenetration module (native, no Warp) | DONE |
| `utils/interpolation.py` — `linear_interpolation_coefficients` | `model::interpolation` / `kernels::math::linear_interpolation` | DONE |
| `utils/kinematics.py` — forward kinematics | `kernels::math::forward_kinematic` | DONE |
| `utils/kinematics.py` — `parallel_forward_kinematic*` | Not ported as separate entry points; the native path is single-threaded and is the target of the threading work in the performance phase | DEFERRED (performance phase) |
| `utils/kinematics.py` — `identity_rotation_like`, `get_kinematic_propagation_fronts` | `kernels::math` (`identity_poses`, propagation ordering inside the kinematic kernels) | DONE (internals) |
| `utils/mesh_utils.py` — `point_to_mesh_distance`, `_and_face`, `_and_face_uvs`, `_and_face_uvs_backward` | `crates/anny-core/src/mesh.rs` BVH: `Bvh::closest` returns distance, face index and barycentrics in one query, so the `_and_face`/`_and_face_uvs` split collapses into one native API. The backward pass is not needed as a separate kernel because the native fitting path is a specialized JVP/VJP. | DONE |
| `utils/obj_utils.py` — `load_obj_file`, `save_obj_file` | `crates/anny-core/src/mesh_io.rs` (OBJ/PLY/STL/glTF/GLB) | DONE |
| `utils/pose.py` — `transfer_pose_parameters` | `crates/anny-core/src/tools.rs::transfer_pose_parameters` | DONE — tests added by this audit |
| `fine_tuning/utils.py` — `find_correspondences`, `load_correspondences`, `make_smpl_target_info`, `make_smplx_target_info` | Correspondences are caller-supplied in the native fitting API; SMPL/SMPL-X target construction in `smpl.rs` | DONE (capability) / EXTERNAL (data) |
| `face_segmentation.py` — `get_face_segmentation_mask` | `assets::AssetStore::segment_faces` | DONE — tests added by this audit |
| `shape_distribution.py` — `get_distribution_params`, `get_torch_distribution`, `ConditionalBetaDistribution`, `SimpleShapeDistribution` | `crates/anny-core/src/distribution.rs` (`ConditionalBetaDistribution`, `SimpleShapeDistribution`, `SampleOptions`) | DONE |
| `paths.py` — `get_anny_root_dir`, cache dir | `assets::AssetStore::new(root)`, `cache.rs` content/config-addressed cache | DONE |
| `paths.py` — `download_noncommercial_data` | Deliberately not ported. Downloading noncommercial SMPL/SMPL-X/correspondence material has no place in a production native stack; user-supplied licensed models are supported instead. | NOT REQUIRED (licensing) |
| `models/model_data.py` — `load_blend_shape`, `load_all_blendshapes`, `load_mesh`, `load_rig`, `build_anny_model_data` | `crates/anny-core/src/import.rs` (native upstream import) + `assets.rs` builders | DONE |
| `models/model_data.py` — `resolve_phenotypes`, `resolve_blendshape_mask` | Parameter/label resolution in `config.rs` (`Selection`, phenotype label handling) and `assets.rs` submodel masks | DONE |
| `models/mesh.py` — `create_fullbody_model`, `create_hand_model`, `create_head_model` | `TopologySpec` (`head`, `hand.L`, `hand.R` …) plus base-vertex/face selection in `assets.rs`; `build_alternative`, `build_soma` | DONE |
| `rig.py` — `create_default_rig`, MakeHuman/Anny/SMPL/SMPL-X rigs | `RigSpec` (`Anny`, `Makehuman`, `Soma`, `Mixamo`, …) + rig construction in `assets.rs` | DONE |
| `torch_compat.py` | Python/PyTorch plumbing with no product meaning | NOT REQUIRED |

## Tutorials

Upstream `tutorials/` are jupytext notebooks. Their native equivalents:

| Upstream tutorial | Native documentation | State |
|---|---|---|
| `alternative_models` | `docs/COMPATIBILITY.md`, `docs/UPSTREAM_AUDIT.md` (this file), `docs/AUTHORING.md` | DONE |
| `keypoints` | `docs/AUTHORING.md` (`KeypointsRegressor`) | DONE |
| `pose_parameterization` | `docs/COMPATIBILITY.md` (the five conventions), `docs/AUTHORING.md` | DONE |
| `pose_transfer` | `docs/AUTHORING.md` pose-transfer section | DONE |
| `shape_parameterization` | `docs/AUTHORING.md` (phenotypes, local changes, interpolation), `docs/COMPATIBILITY.md` | DONE |
| `texture` | `docs/SCENES_AND_MESH_IO.md` (materials, embedded images, UVs) | DONE |

## Upstream tests → native coverage

Upstream test *intent* is mirrored; drop-in compatibility with the Python test files is not a goal.

| Upstream test | Native coverage |
|---|---|
| `test_inverter.py` | `tests/fitting.rs`, `tests/post_gd.rs`, `tests/refinement.rs` |
| `test_degenerate_configuration.py` | `tests/smoke.rs`, `tests/typed_runtime.rs` |
| `test_facial_actions.py` | `tests/smoke.rs`, `tests/typed_runtime.rs` |
| `test_instantiation.py` | `tests/smoke.rs` (incl. legacy/default construction) |
| `test_keypoints.py` | `tests/smoke.rs`, `tests/authoring.rs` |
| `test_kinematics.py` | `tests/smoke.rs`, `tests/typed_runtime.rs` |
| `test_legacy_syntax.py` | `tests/smoke.rs` |
| `test_local_changes.py` | `tests/smoke.rs`, `tests/typed_runtime.rs` |
| `test_makehuman_construction_aliases.py` | `tests/smoke.rs` (MakeHuman rig aliases) |
| `test_pose_parameterization.py` | `tests/typed_runtime.rs` (all five conventions) |
| `test_pose_transfer.py` | `tests/authoring.rs` (added by this audit; identical-rig exactness, missing-bone rejection, rest-mesh rejection, and a real makehuman→anny transfer reproducing the posed mesh) |
| `test_skinning.py`, `test_skinning_weights.py` | `tests/smoke.rs`, `tests/authoring.rs`, `tests/native_import.rs` |
| `test_transforms.py` | `tests/authoring.rs`, `tests/mesh_io.rs` |
| `test_utilities.py` | `tests/mesh_io.rs`, `tests/authoring.rs` |
| `test_soma.py` | `tests/typed_runtime.rs`, parity case `soma` |
| (no upstream test) `face_segmentation` | `tests/authoring.rs::segmentation_selects_disjoint_body_parts` (added by this audit) |

## Genuinely absent

Nothing usable from the pinned upstream lacks a native counterpart. The complete list of upstream
names with no direct native equivalent, and why that is acceptable:

| Upstream | Disposition |
|---|---|
| `paths.download_noncommercial_data` | NOT REQUIRED — licensing boundary. Caller-supplied licensed models only. |
| `parallel_forward_kinematic`, `parallel_forward_kinematic_and_retarget` | DEFERRED — superseded by the native threading work in the performance phase rather than by Python-style parallel-kinematics entry points. |
| `torch_compat.*` (`make_buffer`, device handling, safetensors save/load shims) | NOT REQUIRED — Python plumbing. |
| `warp` mesh utilities used only to express GPU traversal | NOT REQUIRED as Python APIs; semantics live in `mesh.rs` and are re-entered by the WebGPU backend. |
| Python exception text, `torch.Tensor` semantics, `requires_grad`/`backward`, exact PyTorch RNG streams, Gradio as a framework | NOT REQUIRED — explicitly out of scope for a native port. |

## Deliberate behavioural differences

| Surface | Difference | Justification |
|---|---|---|
| `segment_faces` vs `get_face_segmentation_mask` | Native returns the kept face indices; upstream returns a boolean mask over all faces. Same selection. | Index lists are what the native transform/filter operations consume. |
| `segment_faces` pixel indexing | Native clamps both axes to the image bounds (`0..w-1`, `0..h-1`) and uses round-half-even. Upstream clamps `u` by the image *height* and uses round-half-away-from-zero. | The committed `body_parts_segmentation.png` is 1024×1024, so the upstream axis mix-up is inert and both agree on the shipped asset. Not replicated because it would be preserving a latent bug. |
| `point_to_mesh_distance*` | Native BVH returns distance, face and barycentrics in one query rather than three entry points plus a custom backward. | Same semantics, fewer kernels; the native fitting path uses specialized JVP/VJP instead of autograd. |

## Gaps this audit closed

- `transfer_pose_parameters` had no test at all. It now has fast synthetic tests (identical-rig
  exactness within 1e-12, missing-bone rejection, rest-mesh mismatch rejection) and real-data
  ignored tests mirroring upstream `test_pose_transfer.py`, including the makehuman→anny direction
  that has to reproduce the posed mesh.
- `segment_faces` had no test. It now has a real-data disjointness/union test.
- Benchmarks existed only as the `anny benchmark` command, which times a single forward pass in the
  default f64 configuration. `crates/anny-core/benches/runtime.rs` (`cargo bench -p anny-core`) now
  covers the configuration matrix (f64/f32, LBS/DQS, anny/makehuman rigs, all-phenotype and
  local+facial selections), batch generation at 1/10/100 characters, cold prepare, prepared-payload
  reload, and the secondary operations. Both the performance and GPU phases need that spread as a
  baseline.
- Four upstream tutorial workflows (`alternative_models`, `pose_parameterization`, `pose_transfer`,
  `shape_parameterization`) had no discoverable native documentation entry point. See the tutorials
  table above.

## How to re-run the audit

```bash
python3 tools/upstream_audit.py \
  --upstream /home/slurp/Source/Not_Saved/anny \
  --rust .
```

The script only reports name matches; treat its "missing" column as a list of candidates to review by
hand, not as a gap list.
