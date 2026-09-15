# Native authoring

All authoring operations in this document are native Rust. Python is not required to transform, fit, precompute, query, refine or export models.

## Prepared-model transforms

Apply explicit model-data transformations and save a new prepared model:

```sh
./target/release/anny transform \
  --model output/source.safetensors \
  --operations examples/transforms.json \
  --output output/transformed.safetensors
```

The native transform layer keeps dependent topology, UV, bone, weight and orientation data synchronized. Operations reject unsupported/precondition-violating combinations rather than silently leaving stale arrays.

The transform surface covers the useful upstream model-data work: blendshape/face filtering, mesh edits, triangulation, unattached-vertex removal, interpolation/retopology helpers, rig filtering/remapping, skin-weight symmetry/island cleanup/compaction and orientation-related preparation.

## Rig/orientation preprocessing

```sh
./target/release/anny precompute-rig \
  --assets data --rig anny \
  --output output/anny-orientation-cache.safetensors

./target/release/anny precompute-rig \
  --assets data --rig soma \
  --output output/soma-orientation-cache.safetensors

./target/release/anny cache-orientations \
  --assets data \
  --output output/oriented-model.safetensors

./target/release/anny recompute-weights \
  --assets data \
  --output output/cleaned-weights.json
```

These operations are qualified against the committed reference data; see [V1 validation](V1_VALIDATION.md).

## Optional native disk cache

Asset construction can opt into a native content/config-addressed cache:

```sh
./target/release/anny inspect --assets data --cache-dir auto
```

or a caller-selected directory:

```sh
./target/release/anny inspect --assets data --cache-dir /tmp/anny-cache
```

The cache is disabled unless requested. Entries are tied to the source/configuration content and are hash-checked before use.

## Fitting ordinary mesh files

For mesh files, use explicit correspondence semantics:

```sh
./target/release/anny fit \
  --assets data \
  --target target.glb \
  --correspondence index \
  --mesh-fit-options examples/mesh-fit-options.json \
  --output output/fitted.json
```

or initialized local closest-surface fitting:

```sh
./target/release/anny fit \
  --assets data \
  --target target.ply \
  --correspondence closest-surface \
  --mesh-fit-options examples/mesh-fit-options.json \
  --output output/fitted.json
```

Native fitting supports known correspondence, local closest-surface fitting and explicit landmark similarity initialization. The latter two are useful registration tools, but they do not imply globally robust automatic alignment for every unknown/unoriented scan.

## Analytic differentiation and refinement

Native v1 includes specialized analytic derivatives for Anny fitting controls and a native Adam refinement stage.

The shared serialized query API uses the same request definitions from Rust, C/C# and WASM. CLI example:

```sh
./target/release/anny query \
  --model output/model.safetensors \
  --request request.json \
  --output response.json
```

Supported shared operations are:

- `jvp` — directional output derivative for selected parameter directions.
- `vjp` — contract output cotangents back onto a selected Anny control set.
- `refine` — native Adam refinement against target vertices.
- `prior-gradient` — calibrated shape-prior loss and phenotype gradient.
- `motion-resample` — resample a native pose clip.
- `align-landmarks` — solve explicit landmark similarity initialization.
- `measure` — anthropometry.
- `keypoints` — supplied keypoint-regressor query.
- `pose-convert` — convert generated pose representation.
- `sample` / `prior-loss` — calibrated distribution operations.
- `fit` / `fit-mesh` — inverter and ordinary-mesh fitting.
- `collision` — deterministic CPU self-intersection partner query.

This API is deliberately explicit/serializable rather than pretending foreign languages own Rust/PyTorch tensor objects.

### `post_gd`

The inverter's optional `post_gd` path is native. It uses the specialized analytic derivatives plus native Adam and supports the bounded control surface needed by Anny fitting: phenotype logits, pose/root rotation vectors, root translation, local/facial clamps, shared phenotypes and optional calibrated prior regularization.

It is **not** a generic framework-autograd engine and does not promise the identical f32 optimizer trajectory of the upstream PyTorch implementation.

## Motion authoring

Native `PoseClip` support includes validation, interpolation/resampling and retargeting. Export to GLB:

```sh
./target/release/anny motion \
  --assets data \
  --source examples/motion.json \
  --fps 30 \
  --precision f32 \
  --output output/motion.glb
```

NPY/NPZ parsing supports the native AMASS workflow. `amass-fit` requires caller-supplied external SMPL-X/SMPL data/correspondence where NAVER does not commit those licensed assets.

## glTF retained-document authoring

For edits that must preserve authored document state, use `gltf-edit` / `gltf-query` instead of flattening through the geometry-only importer:

```sh
./target/release/anny gltf-edit \
  --source input.glb \
  --operations edits.json \
  --output result.glb

./target/release/anny gltf-query \
  --source result.glb \
  --request query.json \
  --output result.json
```

The retained document layer supports the base-glTF features used by this project, including:

- morph targets / weight animation,
- PNG/JPEG image embedding,
- PBR material data,
- skeletal/TRS animation import and sampling,
- animation editing/query paths,
- output as normal GLB bytes for C/C#/WASM consumers.

Unsupported extensions/compression/codecs are not silently interpreted as equivalent base-glTF data. See [Scenes and mesh I/O](SCENES_AND_MESH_IO.md).

## Portable calibration/keypoint data

```sh
./target/release/anny export-calibration \
  --assets data \
  --output output/calibration.json

./target/release/anny export-keypoints \
  --assets data \
  --output output/keypoints.json
```

These make the relevant secondary data explicit/portable rather than depending on Python objects.

## Pose transfer between rigs

Upstream's `pose_transfer` tutorial moves a posed configuration from one rig to another without
going through Python. The native equivalent is
`anny_core::tools::transfer_pose_parameters(&source, &target, &parameters, mode)`.

The rules are deliberate and match upstream's `test_pose_transfer.py`:

- the target's bone names must all exist in the source rig, otherwise the call errors naming the
  missing bone — never silently fall back to joint indices;
- source and target must share the same rest geometry (checked, not assumed);
- the result is the source pose re-expressed against the target's rest orientations, so a mesh posed
  with the source model and the same mesh posed with the transferred parameters agree. Real-data
  qualification: makehuman→anny over 104 shared bones reproduces the posed mesh to `3.013e-6`
  (upstream asserts `< 1e-4`).

Related workflows:

| Upstream tutorial | Native entry point |
|---|---|
| `pose_parameterization` | the five pose conventions (`PoseParameterization`) plus `anny query` pose-convert requests; `docs/COMPATIBILITY.md` |
| `shape_parameterization` | `Parameters` phenotype kwargs, local changes, `interpolate_model_data` / `interpolate_skinning_weights`, sampling |
| `alternative_models` | `TopologySpec` / `RigSpec` alternates (`soma`, `smpl`, `smplx`, `makehuman`, game-engine variants) and the SMPL-X adapter; `docs/COMPATIBILITY.md`, `docs/NATIVE_IMPORT.md` |
| `texture` | `docs/SCENES_AND_MESH_IO.md` (material/texture authoring and embedding) |
| `keypoints` | `KeypointsRegressor::coco` with an optional explicit label list, used through `Request::Keypoints` by the C/C#/WASM surfaces |

## Benchmark command

```sh
./target/release/anny benchmark \
  --assets data \
  --iterations 20 \
  --warmup 2 \
  --report output/benchmark.json
```

`anny benchmark` times one forward evaluation in the default f64 configuration and writes a JSON
report. For the configuration matrix, batching behaviour, prepare/reload cost and the secondary
operations, use the native benchmark harness:

```sh
cargo bench -p anny-core --bench runtime
```

Both are reproducible CPU timing harnesses, not real-time performance guarantees. Baseline numbers
and what they imply are recorded in `docs/PERFORMANCE.md`.
