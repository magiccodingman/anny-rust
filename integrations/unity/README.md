# Anny for Unity

Native Anny character runtime for Unity: load or build an Anny model, generate a
character into the scene, drive it from phenotype sliders and poses, and bake the
result into ordinary Unity assets.

The package talks to the versioned C ABI in `include/anny.h` through the same
`libanny` that the CLI, the C and .NET samples and the browser build use. There is no
second implementation of the runtime here and no reimplementation of the algorithms in
C#: everything that decides geometry is native code.

## Layout

    package/            the UPM package (com.magiccodingman.anny)
      Runtime/          runtime bindings and integration
      Editor/           baking and authoring tools
      Plugins/x86_64/   libanny.so, staged by tools/build-native.sh
      Tests/            EditMode and PlayMode suites
    project/            host Unity project used to validate the package
    tools/              native build and headless test runners

## Requirements

- Unity 6000.0 or newer (validated on 6000.6.0f1).
- A prepared Anny model. The repository's `output/ci-model.safetensors` is the model the
  suites run against; point tests elsewhere with `ANNY_MODEL`.

## Native library

`tools/build-native.sh` builds `anny-capi` and stages it as
`package/Plugins/x86_64/libanny.so`. The plugin is named so that `[DllImport("anny")]`
resolves on every platform, and the script records the artefact's SHA-256 next to it so a
binary can be traced to the commit that produced it.

## Using it

    AnnyModelAsset asset = ...;               // baked model asset, or
    using AnnyModelF32 runtime = asset.OpenRuntime();

The two supported ways to get a character into a scene:

- **Skinned** (`AnnyUpdateMode.Skinned`) builds a `SkinnedMeshRenderer` with bones, bind
  poses and bone weights. Unity poses the bones; this is the normal Unity path and it
  costs no per-frame native call.
- **Exact** (`AnnyUpdateMode.Exact`) uploads the native evaluated vertices into the mesh
  every update. Nothing about the geometry is left to Unity's skinning, which is what you
  want when the mesh must match a native evaluation bit for bit.

Both modes share the same mesh: per-corner UVs are expanded into distinct mesh vertices
with an explicit corner-to-source map, quads are triangulated, and morph targets from the
model's blendshape stack can be attached as Unity blend shapes.

## Coordinate conventions

Anny is Z-up, right-handed, metres. Unity is Y-up, left-handed, metres. The change of
basis lives in exactly one place, `AnnyRepresentation`, and it is the mapping
`(x, y, z) -> (x, z, -y)`. That mapping is a *proper rotation* (its basis matrix has
determinant +1; it is a quarter turn about X), not a reflection, so face winding is
preserved and triangles are emitted in their original corner order. The suites check this
with the signed volume of the generated mesh rather than trusting the argument.

## Influence order

Unity requires a vertex's bone influences in descending weight order and treats anything
else as an error. Anny does not store them sorted, so the mesh builder sorts each vertex's
influences before writing them, which also makes any truncation drop the smallest weights
rather than whichever happened to come first. Every influence the model assigns is kept by
default; the report records the width, the widest count actually written, and how many
vertices arrived unsorted.

## Running the suites

    tools/run-unity-tests.sh EditMode
    tools/run-unity-tests.sh PlayMode

The runner is deliberately serial: one Editor invocation at a time, results parsed before
the next dispatch. Two overlapping runs compile the same tree twice and make the log
ambiguous about which revision it describes.

## Baking

`Assets > Anny > Bake character...` writes a payload `.bytes` asset, a model asset, a mesh
asset with bind poses and blend shapes, a material, and a prefab whose hierarchy is the
bone rig. The result needs no native plugin at runtime beyond the model payload if you only
use the pre-evaluated mesh.
