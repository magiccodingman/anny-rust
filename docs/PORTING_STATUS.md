# Porting status and remaining work

Baseline: NAVER Anny `81ca83e202273b306205c1cc15f33734be31e48c`.
This is a capability ledger, not an assertion that the entire approved roadmap
has been completed. A compiled adapter or a smoke test is not full qualification.

## Delivered native capabilities

| Area | Current implementation | Evidence / boundary |
| --- | --- | --- |
| Model generation | Morphs, phenotypes, rig/topology variants, five pose conventions, LBS/DQS | Original 23-case forward reference record; not regenerated during this extension |
| Raw upstream data | Restricted `.pth`/`.pt` tensor decoder, YAML, OBJ, compressed targets, native import/manifest verification | All eight repository tensor archives compared against committed conversions; no Python needed |
| Authoring transforms | Synchronized morph/bone/orientation filtering, topology and vertex edits, weight cleanup, rig filtering, interpolation and retopology helpers | Native tests; explicit preconditions in `AUTHORING.md` |
| Precomputation | Anny and SOMA orientation covariance bakes, custom orientation baking, default skin-weight cleanup/export | Compared with committed real-data references; authored SOMA rig/RBF input is reused |
| Mesh files | OBJ, PLY, STL and supported uncompressed triangle glTF/GLB geometry import/export | Native round trips; geometry import bakes skin/morph pose, not full material/animation authoring |
| Scenes | Independent characters, rigged GLB hierarchy/weights/bind transforms/UV seams, supplied skeletal animation frames | Real body and animated synthetic scene passed Khronos validation |
| Fitting | Existing finite-difference/joint registration plus explicit index-correspondence or local closest-surface mesh fitting | Self-mesh round-trip probes, not arbitrary scan-registration equivalence |
| Language access | C/C# and WASM generation, owned GLB/prepared bytes, model transforms, pose transfer and shared secondary queries | Native C lifecycle exercised; C# runtime check is in CI; WASM target compile checked |
| Secondary queries | Measurements, keypoints, sampling/prior, fitting, pose conversion and CPU collision | Some need supplied calibration/weights; not every standalone Rust helper has a binding |
| Caching | Optional content/config-addressed model cache, payload hash checks, atomic publication | Miss/hit, asset invalidation and corruption tests; separate from canonical assets |
| Benchmarks/docs | Explicit CPU benchmark command and authoring, I/O, import and validation documentation | No throughput guarantee; large qualification stays opt-in |

## Still pending in the approved native-completeness work

- A **true f32 evaluation backend**. Existing math is f64; writing f32 glTF
  attributes is serialization, not an f32 runtime. This needs typed buffers,
  numerical kernels, cross-mode tests, and additive typed language interfaces.
- Full AMASS/SMPL-X motion-to-Anny fitting and native motion conversion workflow.
  Exporting already-supplied Anny skeletal poses is implemented, but is not AMASS
  conversion. Third-party licensed model files are neither fetched nor bundled.
- General differentiation and the inverter's optional `post_gd` refinement. The
  existing finite-difference/registration fitter is not a substitute claim.
- Full arbitrary external-mesh registration parity with `mesh_to_params.py`.
  Closest-surface mode currently requires a reasonably aligned input and finds a
  local fit; it is not a complete scan alignment/landmark/robust-loss pipeline.
- Remaining legacy aliases, standalone research workflow coverage, full
  convenience-API coverage across every language, and expanded upstream edge-case
  fixtures. Concrete current transform preconditions remain documented, not
  silently widened.
- Complete morph-channel, texture/PBR material and animation-import authoring.
  Current scene export includes rigging and supplied skeletal animation, while
  ordinary mesh import is geometry-focused.
- Requalification of all original Python-reference cases at the final intended
  v1 commit, plus broader optimizer, collision and actual browser runtime tests.

## Later performance/integration phase

GPU/WebGPU kernels, SIMD optimization, a Unity package, and a complete browser
editor remain separate unfinished work. No GPU, Unity, actual browser-runtime,
or real-time performance claim follows from current compile checks.

## Deliberately not required for the portable product

Python import/drop-in semantics, PyTorch tensor objects or `torch.compile`, exact
Python exception text, PyTorch seed streams and Warp traversal ordering are not
portable runtime requirements. Licensed datasets absent from the upstream repo
remain user-supplied optional inputs. Optional Python reference tooling may stay
in `tools/`; Cargo and the production runtime do not invoke it.

Validation detail: `EXTENSION_VALIDATION.md`, `VALIDATION.md`, and
`COMPATIBILITY.md`. The committed asset tree and the user's additional tools are
preserved; generated caches, meshes and reference-only preparation artifacts are
not part of the normal source payload.
