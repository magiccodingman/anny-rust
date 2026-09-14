# Native authoring and secondary APIs

All commands below are Rust executables. Python is optional reference-generation
software, not an install, build, import, authoring, or execution requirement.
The imported `data/` tree in this branch is ready to use.

## Model transforms

```sh
cargo build --workspace --release --locked
./target/release/anny prepare --assets data --output output/default.safetensors
./target/release/anny transform --model output/default.safetensors \
  --operations examples/transform.json --output output/modified.safetensors
```

`transforms::apply_pipeline` returns a new model; it never mutates a shared
source. Operations are tagged by `op`. Supported operations are `triangulate`,
`edit-mesh`, `filter-faces` (`indices`), `filter-blendshapes` (`labels`),
`remove-unattached-vertices`, `symmetrize-skinning-weights`,
`remove-skinning-islands`, `compact-skinning-weights`, `filter-rig` (`remove`,
`subtree`), `procrustes-orientation`, and `retopology` (`mapping`).

A retopology mapping carries source `indices`, interpolation `weights`, target
`faces`, and optional `template_vertices`, `base_mesh_vertex_indices`,
`texture_coordinates`, and `face_texture_coordinate_indices`. These are Tensor
objects (`shape`, flat row-major `data`, `kind`). The Rust structs are the exact
schema; unknown JSON fields are rejected. Signed local morph pairs must be kept
or removed together. Bone-head/tail/orientation blendshape rows stay synchronized.

Important boundaries:

- `edit-mesh` expects the original MakeHuman vertex numbering. It is not an
  arbitrary topology repair operation.
- Vertex removal composes the original source-index map. It refuses to discard
  nonzero runtime-Procrustes samples rather than silently corrupt orientations.
- Generic rig pruning with runtime-Procrustes or SOMA child-refinement buffers is
  explicitly rejected; convert to an appropriate cached rig first. Bone-dependent
  cached arrays are remapped along with names, hierarchy, and weights.
- Modifying weights does not implicitly redesign a rig's orientation convention.
  Orientation preprocessing is a separate explicit authoring operation.
- Prepared configuration records construction options; transformed `ModelData`
  itself is authoritative. Rebuilding the old config from raw assets does not
  replay a transform pipeline. Save the operation JSON alongside an authored model.

Additional public Rust functions include signed `interpolate_skinning_weights`,
convex `interpolate_model_data` (including face-less point sets),
`retopology_from_mesh`, legacy `apply_procrustes_retopology` with target-surface
sample reprojection, `apply_anny_cached_orientation`, symmetry mapping,
`regress_soma_bone_origins`, and MakeHuman-style `export_weights`.
`assets::apply_soma_rig` and `select_blendshape_rows` remain public native helpers.

## Regenerate preprocessing outputs in Rust

```sh
./target/release/anny precompute-rig --assets data --rig anny \
  --output output/anny-covariance.safetensors
./target/release/anny precompute-rig --assets data --rig soma \
  --output output/soma-covariance.safetensors
./target/release/anny recompute-weights --assets data \
  --output output/weights.default.json
```

The Anny bake supports `weighting` (`skinning`, `skinning-squared`,
`principal-squared`), `aim_weight`, `aim_target` (`tail`, `children`), and
`align_root_with_pelvis` in `--options`. Defaults reproduce the committed preset:
squared skinning, tail aim 0.5, adult reference, and root/pelvis alignment.
For the SOMA bake `--options` accepts `{"threshold":0.01}`.

Both covariance bakes start from geometry/rig assets and do not need an existing
covariance cache. SOMA still uses the **authored, committed** `soma_rig.pt`, including
its RBF and bind-pose data. Reconstructing that authored data from an external
SOMA-X package is not claimed; the package is not a required dependency here.

`cache-orientations` bakes a custom model into a prepared model using an explicit
reference parameter file and the same orientation options. A custom rig without
the Anny pelvis labels should set `align_root_with_pelvis` to `false`.

Precompute commands write only the requested output. They do not overwrite
canonical `data/`, delete source files, or rewrite the source import manifest.
Moving a changed bake into production assets is an explicit authoring decision.

## Optional content-addressed disk cache

```sh
./target/release/anny inspect --assets data --cache-dir output/native-cache
./target/release/anny inspect --assets data --cache-dir output/native-cache
# Second invocation reports a hit on stderr.
```

The default is disabled. `--cache-dir auto` uses `ANNY_CACHE_DIR` or the OS user
cache (`XDG_CACHE_HOME`/`~/.cache`, macOS Library/Caches, Windows LOCALAPPDATA).
Explicit prepared models never consult the filesystem cache.

The key includes a canonical config, upstream revision, schema, native cache
protocol, and content hashes of source assets, including external custom rig or
weight files. Content is hashed during construction/loading, not every frame.
An asset edit invalidates the key even if its timestamp is unchanged. Callers
should keep source assets stable during a build; detected concurrent edits abort
publication. Native protocol v1 is intentionally not Python's cache-key protocol.

Entries contain a payload checksum and length. An incomplete/corrupt cache is an
error, not a silent fallback or permission to delete user files. Remove the named
bad entry explicitly. Publication uses a same-directory temporary file and an
atomic hard link; a filesystem supporting hard links is required for this optional
cache. The cache directory must be outside the source asset tree.

The Rust API is `cache::ModelCache`; custom asset providers can use a `CacheKey`
with their own comprehensive dependency digest. C/C# can build with the optional
cache. Browser hosts continue to load/store prepared byte arrays themselves.

## Fit normal mesh files

```sh
./target/release/anny fit --model output/default.safetensors \
  --target edited-body.obj --correspondence index --output output/fitted.json
./target/release/anny fit --model output/default.safetensors \
  --target scan.ply --correspondence closest-surface \
  --mesh-fit-options fitting-options.json --output output/fitted-surface.json
```

**Index mode** is an explicit assertion that vertices correspond to this model.
A matching count by itself does not establish that assertion. OBJ numbering and
this exporter’s single-character glTF `sourceVertexIndices` metadata allow
consistent UV-seam duplicates to be collapsed. Missing IDs, conflicting duplicate
positions, or incompatible vertex counts are errors. Multi-character scenes are
not treated as one model's vertex correspondence.

**Closest-surface mode** performs a local ICP-style loop: project the current
body to target triangles, run the native fitting step on those correspondences,
and repeat only while the forward surface distance does not regress. The target
must already have approximately compatible pose, scale, and coordinates. A
`target_transform` can explicitly align it. This does not perform global scan
registration or promise the same optimization trajectory as upstream's
`mesh_to_params.py`. Missing limbs, clothing, and severe initial misalignment
require a more specialized fitting workflow.

`MeshFitOptions` includes `outer_iterations`, `inner_iterations`, `tolerance`,
optional `max_distance`, row-major `target_transform`, `initial` (FitOptions), and
`inverter` (InverterOptions). Regular Safetensors targets retain the original
paired-vertex `fit --options ... --inverter-options ...` behavior.

## Same secondary API from Rust, C, C#, and WASM

`operations::Request` / `execute` provide these tagged `operation` requests:

| Operation | Required fields beyond optional `parameters` |
| --- | --- |
| `measure` | none |
| `keypoints` | `regressor`: labels, weights, optional indices |
| `pose-convert` | `mode`: one of the five native pose conventions |
| `sample` | `distribution`, optional `options` |
| `prior-loss` | `distribution`, `phenotypes` |
| `fit` | `target` tensor, optional `setup` and `options` |
| `fit-mesh` | `mesh`, `correspondence`, optional `options` |
| `collision` | optional `group_toes`, `group_eyes`, `group_tongue` |

```sh
./target/release/anny query --model output/default.safetensors \
  --request examples/query-measure.json --output output/measurements.json
./target/release/anny export-calibration --assets data --output output/calibration.json
./target/release/anny export-keypoints --assets data --output output/keypoints.json
```

The last two commands export actual committed calibration/regression data into
portable JSON usable by an in-memory/browser request. They do not require Python.
These are control-plane JSON calls, not a claim of allocation-free per-frame
skinning. Native forward evaluation and tensor views remain separate.

C ABI 1 gained additive functions: `anny_model_query`, `anny_model_transform`,
`anny_model_prepared_bytes`, `anny_model_transfer_pose`, and
`anny_model_build_cached`. Allocated text is released with `anny_string_free`;
bytes with `anny_bytes_free`; new model handles with `anny_model_free`.
Existing ABI functions and tensor-view layout are unchanged.

C# methods: `Query`, `Transform`, `SavePrepared`, `TransferPoseTo`, `FromAssets`.
WASM methods: `query`, `transform`, `prepared_bytes`, `transfer_pose`.
GLB export is also available over all three bindings. A transformed model owns
independent data; it does not invalidate an existing model or borrowed view.

## Small operational timing probe

```sh
./target/release/anny benchmark --model output/default.safetensors \
  --iterations 10 --warmup 1 --report output/timings.json
```

This separates construction/load from repeated CPU-f64 evaluation and records
batch size, raw timing samples, median, p95, OS, and architecture. It is a useful
local timing probe, not a full GPU/browser/convergence qualification campaign.
The default unit suite remains small; real preprocessing tests are explicitly
ignored unless requested:

```sh
cargo test -p anny-core --release --test authoring -- --ignored --nocapture --test-threads=1
```
