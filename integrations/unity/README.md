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

## Modes, and which one is exact

`AnnyUpdateMode.Exact` uploads the *evaluated* vertices into the mesh every update and does no
skinning. What lands in the mesh is the native array, element for element; the tests assert that at
zero tolerance. Use it when the character must match the native result bit for bit.

`AnnyUpdateMode.Skinned` is a normal Unity skinned character: the mesh carries the **bind-pose
geometry** (`rest_vertices`), the bone hierarchy carries the transform, and the renderer skins. This
matters — a skinned mesh is the *input* to skinning, so uploading an already-posed mesh skins the
character a second time. Anny's default evaluation is not a bind pose (`bone_poses` differ from
`rest_bone_poses` by up to 6.8e-02), so the two geometries differ by centimetres and the mistake is
visible rather than subtle.

Unity's skinning is not bit-identical to Anny's, so skinned mode is qualified by tolerance, measured
rather than assumed: baking the skinned character at the evaluated pose reproduces the native
`vertices` array to 6.2e-07 m worst case over 27,436 moved vertices, with `QualitySettings.skinWeights`
set to Unlimited and all nine influences of this rig present. If the active quality level caps
influences, `AnnyCharacter` logs a warning naming the setting and the counts, because a dropped
influence silently changes the result.

The pose session is the fast update path and only covers the skeleton: 0.43 ms against 0.75 ms for a
full evaluation in the same scene. It cannot express phenotype changes, which need a full evaluate.

## Humanoid avatars

`AnnyHumanoid.Build(rig)` returns a Unity humanoid `Avatar` for the anny rig, so an `Animator` can
drive the character from any humanoid clip set. The mapping is in `Runtime/AnnyHumanoid.cs` and is
checked against the native description in `Tests/EditMode/AnnyHumanoidTests.cs`.

Two properties of the rig allowed the mapping without touching the hierarchy:

- the anny `root` bone sits at the pelvis, at the same point as `pelvis.L` and `pelvis.R`, so it is a
  legal `Hips` with both legs and the spine beneath it. `HipsSitsAtThePelvis` asserts that, so a rig
  whose root moved to the floor fails a test instead of quietly producing a wrong avatar;
- the spine is numbered downwards — `spine05` is the lowest and `spine01` the highest — so the slots
  are filled by height: `spine05` -> Spine, `spine04` -> Chest, `spine03` -> UpperChest. `spine01`,
  which is also the parent of the clavicles and the neck, has no slot.
  `SpineSlotsFollowTheHeightNotTheNumber` asserts the heights rather than the names.

Finger order comes from the hand geometry, not from the names: `finger1` is the thumb (its base is the
nearest to the wrist at 0.042 against 0.089-0.100, and it deviates 37.9 degrees from the
middle-finger axis) and `finger5` is the pinky. `FingerOrderMatchesTheHandGeometry` asserts both the
slots and that distance relation.

**What the humanoid definition cannot carry.** 54 of the rig's 104 bones fill the 54 slots.
`AnnyHumanoidReport.UnmappedLabels` names the other 50, which humanoid clips will not drive: the
second bone of every limb segment (`upperarm02`, `lowerarm02`, `upperleg02`, `lowerleg02`), the two
bones above `spine03`, `shoulder01`, `neck02`/`neck03`, the four metacarpals per hand, and thirteen of
the fourteen toe bones per foot. A humanoid playback is therefore an approximation of Anny's pose
rather than a reproduction of it, and Anny's own baked clips remain the exact path. For the prepared
model the editor reports `mapped 54/104 bones onto 54 humanoid slots, missing 0 required, 50 bones
left outside the humanoid definition; valid=True human=True`.

## Player builds

```bash
ANNY_MODEL=/path/to/model.safetensors bash tools/build-players.sh
```

Builds the Linux Mono and Linux IL2CPP players into `player-mono/` and `player-il2cpp/` (both
git-ignored), then runs each one. This is the only surface that exercises the plugin outside the
editor: the editor always runs Mono, so IL2CPP marshalling, `SafeHandle` release and the tensor
readers are only proven in a player.

The scene is generated by the build rather than hand-authored (`Assets/Editor/PlayerBuild.cs`), so
it contains exactly what it says: a character host with generation on awake disabled.
`Assets/AnnyPlayer/PlayerSmoke.cs` runs inside the player, reads the model from `ANNY_MODEL`,
generates the character, asserts the mesh, influence width, skeleton and renderer, prints its lines
and exits with a status code derived from that result. A player that compiles but cannot generate a
character, or that counts its updates wrongly, fails the run.

Both backends currently report:

```
ANNY-PLAYER-SMOKE ok vertices=13718 meshverts=82260 tris=27420 bones=104 influences=9 volume=0.05101393 runtime=mono
ANNY-PLAYER-SMOKE ok vertices=13718 meshverts=82260 tris=27420 bones=104 influences=9 volume=0.05101392 runtime=il2cpp
ANNY-PLAYER-PERF ok runtime=mono iterations=20 phenotype_ms=0.561/0.57 pose_ms=0.323/0.383 updates=40
ANNY-PLAYER-PERF ok runtime=il2cpp iterations=20 phenotype_ms=0.555/0.698 pose_ms=0.309/0.404 updates=40
```

The perf phase measures what a game actually pays per update: 20 phenotype changes (`SetPhenotype` + `Apply`,
the full re-evaluation path) and 20 pose-only updates through the native session, reporting the fastest and the
median update in milliseconds. The median is **0.57 ms (Mono) / 0.70 ms (IL2CPP)** for a phenotype change and
**0.38 ms / 0.40 ms** for a session pose update — under 3% of a 60 fps frame either way. The ~0.19 ms the session
saves is the native evaluation the runtime documents, so what is left is the Unity-side push, and IL2CPP tracking
Mono says marshalling is not a cliff.

The phase also checks `AnnyCharacter.Evaluations`, which counts one update per call (`Generate` counts as the
first, then one per `Apply` or session pose update) and fails unless the delta is exactly two per iteration —
that is what makes "the session was reused" evidence rather than an assumption.

The script waits for the editor process to exit between runs. Starting the next run too early lets
Unity's script compiler backend abort with `Scripts have compiler errors`, which is an
infrastructure artifact rather than a code error.

## Editor controls

`AnnyCharacterEditor` replaces the default inspector for a character: generate and regenerate from edit
mode, the phenotype sliders by label (driving `SetPhenotype` + `Apply`, so the sliders use the same path
a game does), a result box reporting what the mesh builder actually produced — including influences lost
to the cap, which is the failure mode that produces a plausible-looking character rather than an
exception — and preset capture/apply.

`AnnyPreset` is an ordinary asset holding mode, precision, sliders and mesh options. The test that matters
is that applying a captured preset reproduces the source geometry **exactly** (zero tolerance, element-wise
over 27,436 vertices), and that the preset survives `AssetDatabase` serialization.

`AnnyBakeWindow` (`Anny / Bake prepared payload...`) bakes a prepared payload into ordinary Unity assets
and reports where they went.

## Physics

`AnnyMeshCollider` points a `MeshCollider` at the character's generated mesh and re-cooks it when the
geometry changes. In Skinned mode that is the bind-pose surface; in Exact mode it follows the posed
surface. See `docs/VALIDATION.md` for the measured agreement.
