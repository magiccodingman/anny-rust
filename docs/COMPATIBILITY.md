# Compatibility boundary

Reference: `naver/anny@81ca83e202273b306205c1cc15f33734be31e48c`, data version 11.
This is a semantic native port, not a Python import-compatible package.

## Implemented and exercised against the real reference

- Raw OBJ, target/target.gz, rig/weight JSON, face targets and segmentation loading.
- Native restricted .pth/.pt decoding and YAML loading/conversion; no Python needed for import or runtime.
  See `NATIVE_IMPORT.md` for supported serialization and limits.
- ModelData Safetensors loading/saving, configuration stored in prepared models.
- Six default / eleven all phenotype inputs, source anchors including age -1/3,
  ancestry-weight normalization/fallback, optional extrapolation, positive/negative
  local morph pairs, facial actions, source ordering, named and stacked inputs.
- Anny and MakeHuman rigs; cmu_mb, game_engine, mixamo; bone modifiers, pruning,
  reparenting, weight aggregation; head and left/right hand submodels.
- Tail/blender, cached covariance and Procrustes rest orientations; SOMA refinement.
- All five pose conventions, root/reference orientation behavior, batch broadcasting.
- Linear blend and dual-quaternion skinning, including upstream quaternion conventions.
- Native triangle/quad topology, shortest-diagonal triangulation, edits, UV indexing,
  pruning, notoes and SOMA retopology, SOMA rig with SOMA and Anny topology.
- Python-reference comparisons include exact packed bone indices and float-tolerant
  weights, not merely visually similar vertices.

The 23 fixture cases are listed in `tools/export_reference.py`. These are fixed
regression probes, not exhaustive coverage of every input combination. None of
the full/reference cases is replaced with a synthetic body or placeholder mesh.

## Implemented secondary APIs, more limited validation

| Surface | What is present | Validation / limit |
| --- | --- | --- |
| Anthropometry | Height, waist loop, volume, source density-derived mass and BMI | Native computations; only supported topology/waist vertices accepted |
| Keypoints | COCO asset loader; dense/sparse regressors | Synthetic dense/sparse consistency test |
| Pose transfer | Source/target rest-orientation correction and name matching | Core convention round trips; broader rig-transfer cases remain useful additions |
| Shape distribution | Age mapping, age-conditional beta distributions, mixture prior, sampling | Synthetic distribution tests plus real calibration import |
| AnnyInverter | Joint registration, finite-difference phenotype fitting, shared parameters, multistart and regularized solves | Synthetic rigid-fit smoke; not full iterative Python trajectory parity |
| Collision | CPU BVH triangle SAT with upstream exclusion categories | Basic geometry helpers tested; not a validated GPU collision substitute |
| SMPL / SMPL-X | Native forward adapter for explicitly exported licensed data, shape/expression/pose correctives, hand PCA/means | Compiles; no licensed real-model fixtures were available in this environment |
| C ABI | Opaque ownership, status/errors, model load/build/evaluate and array views | Real C executable generated/evaluated a full model |
| C# example | .NET 10 P/Invoke/SafeHandle wrapper and managed output copies | Build in CI; not a Unity integration package |
| WASM | In-memory prepared model, same geometry core, typed-array outputs | Target compilation checked; browser UI/performance not independently qualified |

## Native authoring extension

See [AUTHORING.md](AUTHORING.md) for transforms, covariance and skin-weight bakes,
checksummed caching, ordinary-mesh fitting, and the shared secondary binding API.
See [SCENES_AND_MESH_IO.md](SCENES_AND_MESH_IO.md) for rigged GLB scenes and formats.
Native `.pth`/`.pt` import replaces the previously required conversion interpreter.
These additions do not imply completion of the remaining boundaries below.

## Explicit differences and unfinished surfaces

1. **Automatic differentiation, PyTorch tensor APIs, Warp/CUDA/GPU acceleration,
   torch.compile and the Gradio/Jupyter UI are not ported.** This is the agreed
   dependency-free runtime boundary, not a replacement PyTorch ecosystem.
2. The inverter implements its finite-difference / registration baseline, **not
   optional `post_gd` Adam/autograd refinement**. Unknown configuration keys are
   rejected rather than silently pretending this mode ran. Its f64 solver can
   follow a different optimization trajectory from upstream's f32 implementation.
3. The shape sampler preserves distributions and interpolation semantics but
   uses SplitMix64 / native gamma-beta sampling. Seeded sequences are not PyTorch
   sequences. Samples are not treated as reference parity fixtures.
4. Collision partner selection among several intersections is deterministic on
   CPU, rather than reproducing GPU traversal/order. GPU partner IDs and gradients
   are not claimed to match.
5. Detached, unweighted helper vertices retained in a `*-full` topology are
   root-bound instead of carrying upstream NaN weights. Referenced default
   topology is unchanged. Do not call this intentional hardening byte-identical
   behavior for every unused helper vertex.
6. ModelData transforms, skin-weight cleanup, both covariance bakes, and a local
   closest-surface fitting workflow are now native authoring tools. External
   AMASS/SMPL fitting workflows, global scan registration, plotting/comparison
   programs, and authored SOMA-X RBF generation are not fully replicated. Generic
   rig pruning still explicitly rejects runtime-Procrustes/SOMA-refiner data until
   an appropriate cached rig is prepared. See the authoring documentation.
7. SMPL/SMPL-X licensed models and noncommercial correspondence maps are not
   downloaded or bundled. `tools/export_smpl.py` is an explicit offline preparation
   bridge. Real asset validation remains pending; the adapter is not evidence of
   full third-party library compatibility.
8. C API ABI 1 and WASM now also expose secondary requests, model transforms,
   prepared bytes, pose transfer, and GLB export. Filesystem-specific import and
   preprocessing orchestration remains native CLI/Rust tooling; browser hosts
   provide bytes themselves. A dedicated Unity package is not supplied.
9. Internal arrays are f64 reference arrays and no unsafe code is permitted in
   `anny-core`. There is no promise of float16/float32 output-type parity, SIMD/GPU
   throughput, identical Python cache filenames, or identical exception wording.
   Source-specific f32 projection and rig-roll steps are deliberately reproduced.
10. `prepare` remains explicit. Optional automatic native disk caching now exists
    with content-addressed keys and checksum verification; Python's exact cache
    directory/hash protocol is not replicated. Corrupt/incompatible caches fail.

## Numerical details deliberately preserved

- Flat matrices/vertices use row-major serialization; column vectors for transforms.
- Upstream coordinates remain meters, Z-up. No engine-dependent axis swapping.
- Rig roll matrices deliberately calculate through f32 quaternions before widening,
  matching upstream's source-data path.
- DQS retains upstream's particular dual part and sign-selection convention.
- The tail-orientation degeneracy uses upstream's 180-degree X fallback rather
  than replacing it with a theoretically nicer rotation.
- Closest-surface retopology uses a CPU port of Warp's float32 closest-triangle
  routine and 16-bucket SAH tree. Its traversal and partition tie order matter for
  reproducing packed skin weights on coincident triangles.
- Array ordering, source labels and UV connectivity remain observable API contracts.

This file is intentionally a boundary, not a roadmap claiming unimplemented code
exists. Subsequent PRs can extend these surfaces without adding Python back into
the native generation runtime.
