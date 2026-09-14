# anny-rust

Native Rust implementation of NAVER Anny pinned to
`naver/anny@81ca83e202273b306205c1cc15f33734be31e48c` (ModelData schema 11).

**Normal build/runtime use is Python-free.** The repository includes the pinned asset payload and can also import the upstream data formats natively. Python helpers under `tools/` are optional reference/qualification utilities only.

Native v1 covers body generation, true f64/f32 evaluation, authoring transforms, fitting/refinement, motion, rigged glTF/GLB, retained glTF materials/morphs/animation, and Rust/C/C#/WASM access. Fresh NAVER reference qualification passes 23/23 cases; see [V1 validation](docs/V1_VALIDATION.md) and [compatibility](docs/COMPATIBILITY.md).

## Workspace

| Crate | Purpose |
| --- | --- |
| `anny-core` | Native assets/model math, f64/f32 evaluation, fitting/differentiation, motion, mesh/glTF authoring and utilities |
| `anny-cli` | Preparation, generation, conversion, fitting, motion/AMASS and authoring commands |
| `anny-capi` | Stable native C ABI with typed f64/f32 and serialized secondary operations |
| `anny-wasm` | Browser/WASM wrapper with owned typed-array results and shared query/authoring operations |

Rust 1.90+ is required.

## Build

```sh
cargo build --workspace --release --locked
cargo test --workspace --locked
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
```

No Python, PyTorch, LibTorch, CUDA or Warp runtime is required.

## Generate characters

Use committed assets directly:

```sh
./target/release/anny inspect --assets data

./target/release/anny generate \
  --assets data \
  --config examples/all.json \
  --params examples/character.json \
  --precision f32 \
  --mesh output/character.glb \
  --rigged true \
  --output output/character.safetensors
```

`--precision f64` is the reference/high-precision path; `--precision f32` is a genuine single-precision evaluator using the same equations, not an output-only cast.

Anny coordinates remain **meters and Z-up**. Engine-specific axis/handedness conversion belongs in the integration layer rather than the parity model.

## Prepare once, evaluate many times

```sh
./target/release/anny prepare \
  --assets data \
  --config examples/all.json \
  --precision f32 \
  --output output/anny-all-f32.safetensors

./target/release/anny generate \
  --model output/anny-all-f32.safetensors \
  --params examples/character.json \
  --mesh output/from-prepared.glb \
  --rigged true
```

Prepared models keep model configuration/static data while phenotype, local/facial controls and pose remain dynamic. Optional content/config-addressed caching is also available for asset construction.

## Native import of an untouched upstream checkout

Normal users do not need this because `data/` is already committed. Maintainers can import a fresh pinned checkout without Python:

```sh
./target/release/anny import-upstream \
  --source /path/to/naver/anny \
  --destination output/imported-data

./target/release/anny verify-assets --assets output/imported-data
```

The importer understands the bounded `.pth/.pt` tensor-archive subset actually used by the pinned repository, plus YAML/OBJ/targets and associated asset metadata. It does not execute arbitrary pickle globals.

## Fitting and native refinement

Basic fitting:

```sh
./target/release/anny fit \
  --assets data \
  --target output/character.safetensors \
  --options examples/fit-options.json \
  --output output/fitted.json
```

Native v1 includes:

- known-correspondence fitting,
- initialized local closest-surface fitting,
- explicit landmark similarity initialization,
- specialized analytic JVP/VJP directions,
- native Adam refinement,
- optional inverter `post_gd`,
- phenotype logit constraints,
- root translation/rotation-vector refinement,
- local/facial clamps,
- shared phenotype fitting and optional calibrated-prior regularization.

This is not a generic PyTorch-style autograd system and is not advertised as globally robust automatic registration for every arbitrary unaligned scan.

## Motion and AMASS

Native pose clips can be loaded/resampled/retargeted and exported as animation:

```sh
./target/release/anny motion \
  --assets data \
  --source examples/motion.json \
  --fps 30 \
  --precision f32 \
  --output output/motion.glb
```

AMASS/SMPL-X arrays can be inspected and a supplied source model fitted:

```sh
./target/release/anny amass-inspect \
  --source sequence.npz \
  --output output/amass-info.json

./target/release/anny amass-fit \
  --assets data \
  --source sequence.npz \
  --source-model /path/to/supplied-smplx.safetensors \
  --mapping explicit-map.json \
  --options fit.json \
  --output output/amass-fitted.glb \
  --report output/amass-report.json
```

SMPL/SMPL-X/AMASS assets that are not committed by NAVER are not downloaded or bundled. The native baseline therefore requires the caller to supply legally obtained source data/correspondence where needed.

## Mesh and glTF/GLB authoring

Native interchange includes OBJ, PLY, STL and the documented glTF/GLB subset. Rigged GLB export includes skeleton hierarchy, skinning, bind transforms, UV seam handling and supplied skeletal animation.

A retained `GltfAsset` document layer additionally supports:

- morph-target channels,
- embedded PNG/JPEG textures,
- PBR materials,
- animation import/sampling,
- byte-oriented edit/query operations shared across native language boundaries.

Use retained-document operations when materials/morphs/animation need to survive authoring. Geometry-only import intentionally returns flattened mesh geometry rather than pretending to round-trip every document object/extension.

See [Scenes and mesh I/O](docs/SCENES_AND_MESH_IO.md) and [Authoring](docs/AUTHORING.md).

## C / C++ and C#

The stable C contract is [include/anny.h](include/anny.h), ABI version 1.

```sh
cargo build --release -p anny-capi --locked
mkdir -p output
cc -std=c11 -Wall -Wextra -Werror -Iinclude examples/c_smoke.c \
  -Ltarget/release -lanny_capi -Wl,-rpath,"$PWD/target/release" \
  -o output/c-smoke
./output/c-smoke data
```

The C ABI uses opaque ownership/status/error handling and typed f64/f32 views. Shared serialized query operations expose measurements, keypoints, fitting/refinement, JVP/VJP, transforms and related secondary functionality.

The .NET 10 example wraps ownership with SafeHandle and exercises actual native calls:

```sh
./target/release/anny prepare --assets data --output output/anny-default.safetensors
LD_LIBRARY_PATH="$PWD/target/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  dotnet run --project examples/csharp --configuration Release -- \
  "$PWD/output/anny-default.safetensors"
```

This managed layer is **not yet the deferred full Unity Runtime/Editor package**.

## WebAssembly

The core loads prepared model bytes in-memory and returns owned JS typed arrays.

```sh
rustup target add wasm32-unknown-unknown
cargo check -p anny-wasm --target wasm32-unknown-unknown --locked
```

An actual Chromium qualification has exercised the real model through WASM, including f64/f32 evaluation, measurements, JVP, an Adam refinement step, rigged GLB, texture/morph authoring and animation import/sampling. See [V1 validation](docs/V1_VALIDATION.md).

That proves the portable browser runtime; it is **not** the still-deferred complete browser character-editor UI or WebGPU backend.

## Reference qualification

Optional developer-only Python reference generation remains available:

```sh
PYTHONPATH=/path/to/naver/anny/src \
  python tools/export_reference.py --output output/fixtures --cases all-cases

python tools/check_parity.py \
  --assets data \
  --fixtures output/fixtures \
  --summary output/parity-summary.json
```

Fresh pinned-source qualification passes **23/23** cases at `atol=1e-6`, `rtol=0`, with integer arrays/labels exact. This is strong regression evidence, not a mathematical proof over every possible continuous input.

## What is intentionally still later work

Native v1 establishes portable correctness and authoring. The next project phase is intentionally separate:

1. **GPU/WebGPU backend**.
2. **SIMD tuning**.
3. **Unity Runtime/Editor package**.
4. **Complete browser character editor**.
5. **Serious performance optimization** (profiling, reusable workspaces, allocation/layout tuning, incremental updates, batching/threading and load-time work).

See [Porting status](docs/PORTING_STATUS.md) for precise supported/deferred boundaries.

## Licensing

The port preserves NAVER's Apache-2.0 attribution. MakeHuman/MPFB2 and Face Units assets carry their accompanying CC0 notices; SOMA assets carry their accompanying Apache-2.0 notices. NVIDIA Warp-derived compatibility code remains Apache-2.0 attributed in [NOTICE](NOTICE). SMPL/SMPL-X/AMASS data absent from upstream remains separately licensed, opt-in caller data. See [LICENSE](LICENSE).
