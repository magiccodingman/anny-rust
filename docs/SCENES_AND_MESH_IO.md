# Native scenes and mesh interchange

All operations in this document run in Rust. No Python interpreter, Trimesh,
Blender process, GPU or network service is required.

## Generate one character or a lineup

```sh
cargo build --workspace --release --locked
./target/release/anny generate --assets data --config examples/all.json \
  --params examples/character.json --mesh output/character.glb --rigged true
./target/release/anny lineup --assets data --config examples/all.json \
  --params examples/lineup.json --output output/lineup.glb
./target/release/anny generate --assets data --mesh output/character.ply
./target/release/anny mesh-convert --source output/character.glb --destination output/surface.obj
```

Create the destination directory first. `generate`/`lineup`/`mesh-convert` also create
parent directories. `--mesh` accepts GLB, embedded-buffer glTF, OBJ, PLY and STL.
`--obj` remains supported. A lineup JSON is an array of `{parameters, export}`
objects; export options are `name`, `translation`, `color`, `rigged`, `batch_index`.
Each character is a separate mesh/node/skeleton, not a concatenated mesh.

## Static versus rigged

The default is a static mesh with its shape, expression and pose baked. Rigged
GLB/glTF contains the shaped rest mesh, complete joint hierarchy, inverse bind
matrices and current bone pose. Every positive skin influence is retained across
JOINTS_n/WEIGHTS_n sets; it is not silently reduced to four weights. Zero-weight
padding indices are canonicalized to zero for interchange.

Standard glTF skinning is LBS. A request to export a DQS model as a rigged glTF
returns an error; static DQS export remains available and bakes the actual shape.
Bone transforms and scene placements must be rigid; unsupported shear/scale is
rejected instead of silently converted into a different rotation.

UV seams are split as required by glTF's single vertex indexing. A
`sourceVertexIndices` extras array records their relationship to the original mesh.
The original Anny model and numerical output vertex ordering are unchanged.
Normals are area-weighted; unused/degenerate normals receive a unit fallback.
UV V coordinates are converted from Anny/OBJ's bottom-origin to glTF's top-origin.
Material color, roughness, alpha, and double-sided state are written. Arbitrary
upstream Blender material node graphs are not translated.

## Coordinate system

The core stays Z-up, in meters. glTF files carry a -90 degree X scene-root rotation
to standard Y-up. Joint roots carry that conversion too; skinned mesh nodes remain
at the scene root, because glTF ignores mesh-node transforms when skinning.
The geometry importer converts glTF world-space output back to Z-up meters.
OBJ/PLY/STL are read/written in Z-up meters. STL itself cannot record these units.

## Rust animation and byte APIs

`scene::Scene::add_character` returns an object index. `Scene::add_animation`
accepts strictly increasing sample times and absolute bone poses for that object's
rig. Export creates ordinary glTF translation/quaternion animation channels with
continuous quaternion signs. Shape is fixed for the clip; animated phenotype or
facial blendshape channels are not implemented in this slice.

`Scene::to_glb()` / `to_gltf()` return bytes without filesystem use. C exposes
`anny_model_export_glb` plus an owned `AnnyBytes` handle. The C# example adds
`AnnyModel.ExportGlb()`. WASM exposes `AnnyModel.export_glb()` as an owned Uint8Array.
Free C byte handles with `anny_bytes_free`, never with another allocator.

## Geometry import boundary

`mesh_io::load`/`from_bytes` read:

- OBJ triangles/quads, negative indices and multiple object records;
- ASCII, binary little-endian and binary big-endian PLY, skipping extra properties;
- ASCII and binary STL (coincident vertices welded by exact coordinate values);
- glTF 2.0/GLB triangle primitives, strided/sparse attributes, embedded or local
  relative buffers, node transforms, default morph weights and default skin pose.

This is a geometry reader for fitting/conversion, not a lossless whole-scene
editor. glTF import flattens the selected default scene; animations are not sampled,
materials/textures are not preserved, and compressed/required extensions fail.
PLY/STL export is geometry-only. Arbitrary polygons, Draco/meshopt and external
network buffer URLs are not accepted. External buffer paths cannot escape the
source document directory, including through symlinks. In-memory glTF requires
embedded buffers. Files/combined buffers are capped at 512 MiB.

A different file format does not establish fitting correspondences. Do not assume
an arbitrary scan's vertex i is Anny vertex i just because both load successfully.

## Checks

Native tests cover file round trips, multiple OBJ objects, UV seam mapping,
GLB/embedded-glTF skinned pose and axis round trips, animation layout and invalid
inputs. Khronos glTF Validator 2.0.0-dev.3.10 reported zero errors and zero warnings
for both a two-character animated synthetic scene and a real 104-bone Anny GLB.
The existing numerical model implementation was not changed by these exporters.
