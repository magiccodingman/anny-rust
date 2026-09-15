# Scenes and mesh I/O

Anny-Rust has two intentionally different 3D-data surfaces:

1. **Geometry/scene interchange** — load/save portable mesh geometry and emit rigged character scenes.
2. **Retained glTF document authoring** — preserve/edit/query materials, images, morph targets and animations without flattening them to a bare mesh.

Use the surface that matches the job; geometry import intentionally does not pretend to preserve every glTF document object.

## Geometry formats

Native geometry I/O supports:

- OBJ
- PLY
- STL
- the documented uncompressed triangle glTF/GLB subset

Example conversion:

```sh
./target/release/anny mesh-convert \
  --source input.obj \
  --destination output.ply
```

OBJ preserves separate UV indexing where the format permits it. Anny coordinates remain meters and Z-up internally.

The geometry-only glTF/GLB loader resolves the supported accessors/transforms into a `SurfaceMesh`. That is useful for fitting/conversion, but authored materials, animation clips and morph-channel document structure should be handled through `GltfAsset` when round-trip preservation matters.

## Character GLB export

Generate a real rigged character:

```sh
./target/release/anny generate \
  --assets data \
  --config examples/all.json \
  --params examples/character.json \
  --precision f32 \
  --mesh output/character.glb \
  --rigged true
```

Rigged LBS GLB output includes:

- separate mesh and skeleton nodes,
- bone hierarchy,
- positive skin influences,
- inverse bind transforms,
- current pose,
- UV seam splitting with source-vertex recovery metadata,
- supplied skeletal animation where requested.

DQS deformation can be exported as baked geometry, but it is not falsely labeled as ordinary glTF linear skinning.

## Multi-character scenes

`Scene` supports independent character instances/transforms in one GLB. CLI example:

```sh
./target/release/anny lineup \
  --assets data \
  --config examples/all.json \
  --params examples/lineup.json \
  --output output/lineup.glb
```

Each character keeps its own rig/mesh nodes rather than being collapsed into one synthetic skin.

## Skeletal motion

Native `PoseClip` sequences can be interpolated/resampled and exported:

```sh
./target/release/anny motion \
  --assets data \
  --source examples/motion.json \
  --fps 30 \
  --precision f32 \
  --output output/motion.glb
```

Pose data is explicit about Anny pose parameterization; pose values are not silently reinterpreted between incompatible conventions.

## Retained glTF authoring

`GltfAsset` keeps a base-glTF document plus its buffers/images so higher-level authoring does not need to flatten through `SurfaceMesh`.

Supported native-v1 authoring includes:

- morph-target channels,
- morph weight animation,
- skeletal/TRS animation import and sampling,
- PNG/JPEG embedding,
- PBR material representation,
- material/morph/animation edit/query operations,
- GLB byte output usable from Rust, C/C# and WASM.

CLI:

```sh
./target/release/anny gltf-edit \
  --source input.glb \
  --operations edits.json \
  --output output/authored.glb

./target/release/anny gltf-query \
  --source output/authored.glb \
  --request query.json \
  --output output/query.json
```

The animation loader validates accessor type, normalization and time bounds. Legal normalized integer rotation/weight channels are supported where base glTF permits them. Quaternion sampling normalizes results and uses shortest-path interpolation for linear rotation channels.

## Validation

Real generated Anny GLBs and synthetic/multi-character animation scenes have been checked with Khronos glTF Validator. An actual Chromium qualification also authored a rigged/morphed/textured animated GLB which validated with **0 errors and 0 warnings**.

See [V1 validation](V1_VALIDATION.md) for exact qualification evidence.

## Explicit limitations

Native v1 does not claim every possible glTF extension or compressed asset format. In particular:

- unsupported vendor/extensions/codecs are not silently accepted as equivalent base glTF,
- geometry-only import does not retain authored PBR/animation/morph-document state,
- the complete browser editor UI is not part of this layer,
- WebGPU acceleration is not implemented yet,
- Unity-specific scene/renderer packaging remains a later phase.

These boundaries are intentional; they keep the core portable rather than pulling a large 3D editor/runtime dependency stack into the Rust model library.
