# anny-rust

**Native extension:** this branch now includes native tensor import, rigged GLB/glTF
scenes and animation, OBJ/PLY/STL/glTF geometry I/O, authoring transforms, covariance
and skin-weight preprocessing, optional checksummed caching, normal-mesh fitting,
and secondary C/C#/WASM APIs. See [Authoring](docs/AUTHORING.md) and
[Scenes and mesh I/O](docs/SCENES_AND_MESH_IO.md).

A normal checkout uses the committed `data/` directly: **no Python installation
is needed**. The older Python helpers are optional upstream-reference utilities.


**The data payload is now included. No Python setup is needed to build or generate.**
Native import from untouched upstream tensor archives is also available; see
[Native import](docs/NATIVE_IMPORT.md). The older Python importer below is an
optional developer/reference path, not a prerequisite for using this repository.


A native Rust implementation of the Anny human-body geometry runtime, pinned to
[NAVER Anny](https://github.com/naver/anny) revision
`81ca83e202273b306205c1cc15f33734be31e48c` (ModelData schema 11).

The runtime does not embed Python, PyTorch, LibTorch, CUDA, or NVIDIA Warp.
It reads the original mesh, morph and rig assets and generates meshes, skeletons,
and poses in Rust. A one-time import tool converts Python-specific tensor files
to portable Safetensors without changing the original checkout.

**Status:** native generation has passed 23 Python-reference configurations at
`atol=1e-6`, `rtol=0`, including exact topology, UV indices, packed skinning indices,
and labels. This is not a claim that every Python research helper, gradient API,
or optional dependency has been reproduced. See [the compatibility matrix](docs/COMPATIBILITY.md)
and [recorded validation](docs/VALIDATION.md) for the precise boundary.

## Workspace

| Crate | Purpose |
| --- | --- |
| `anny-core` | Asset loaders, model construction, phenotype/local/facial blending, rest skeletons, five pose conventions, LBS/DQS, topology/rig transformations, fitting and geometry utilities |
| `anny-cli` | `anny` command: prepare, generate, inspect, compare, measure, sample, fit |
| `anny-capi` | Shared/static native library, opaque handles and a versioned C interface |
| `anny-wasm` | Browser wrapper around the same runtime with owned typed-array results |

Rust 1.90 or later is required. All math is CPU reference math, primarily float64;
this first implementation prioritizes numerical compatibility, not a GPU or
real-time performance claim. Large prepared model files and working buffers must
be budgeted explicitly when embedding in a game or browser.

## Build without assets

```sh
cargo build --workspace --release --locked
cargo test --workspace --locked
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
```

The small test suite uses synthetic data and does not need Anny's asset payload,
Python, a GPU, network downloads at runtime, or a full benchmark service.

## Import an untouched upstream checkout (optional)

The asset payload is already committed; skip this for normal use. To import a
fresh copy using only Rust, build the CLI and use a **new destination**:

```sh
cargo build --release -p anny-cli --locked
./target/release/anny import-upstream \
  --source /home/slurp/Source/Not_Saved/anny \
  --destination output/imported-data
./target/release/anny verify-assets --assets output/imported-data
```

The native importer copies the complete `src/anny/data/` hierarchy, converts the
restricted tensor-archive formats actually used by the pinned source, converts
YAML metadata, and writes a source/hash manifest. It never executes pickle globals
or starts Python. See [Native import](docs/NATIVE_IMPORT.md) for its supported
format subset and errors. The source checkout is read-only.

The expected revision is `81ca83e202273b306205c1cc15f33734be31e48c`. Do not reset
an original checkout containing edits. A detached worktree can supply the pinned
source without disturbing it. External SMPL/SMPL-X downloads are neither requested
nor bundled.

## Generate a character

```sh
cargo run --release -p anny-cli --locked -- inspect --assets data
cargo run --release -p anny-cli --locked -- generate \
  --assets data --config examples/all.json --params examples/character.json \
  --obj output/character.obj --output output/character.safetensors
```

Import `output/character.obj` into Blender. The mesh preserves upstream vertex
and face ordering, separate UV indices, **meters and Z-up**. Coordinate conversion
for a Y-up or left-handed game engine belongs in its integration wrapper; it is
not silently applied to the parity implementation.

`examples/all.json` enables every phenotype, local morph and facial action that
upstream exposes. It does not invent new controls or bypass upstream filtering.
Use `inspect --config examples/all.json` to enumerate the actual labels.

## Prepare once, evaluate many times

```sh
cargo run --release -p anny-cli --locked -- prepare \
  --assets data --config examples/all.json --output output/anny-all.safetensors
cargo run --release -p anny-cli --locked -- generate \
  --model output/anny-all.safetensors --params examples/character.json \
  --obj output/from-cache.obj
```

A prepared model carries its configuration and precomputed ModelData. Loading it
bypasses raw asset parsing. Shape, expression and pose parameters remain dynamic;
preparing a model does **not** freeze one character. Reuse the loaded model for
many evaluations. The CLI deliberately does not hide a disk cache; use `prepare`
explicitly so the build/preparation phase is visible and reproducible.

Prepared files can be substantially larger than the compressed source assets.
Keep them under `output/` or in your engine's asset packaging, **not in this PR's
normal Git asset commit**.

## Native Rust use

```rust,no_run
use anny_core::{assets::AssetStore, AnnyConfig, Parameters};

fn main() -> anny_core::Result<()> {
    let model = AssetStore::new("data").build(&AnnyConfig::default())?;
    let input: Parameters = serde_json::from_str(
        r#"{"phenotype_kwargs":{"height":0.65,"weight":0.4}}"#
    )?;
    let result = model.forward(&input)?;
    let vertices = result.get("vertices")?; // shape [batch, vertex, xyz]
    println!("{:?}", vertices.shape);
    Ok(())
}
```

Named parameter dictionaries accept scalars or batch vectors; stacked arrays use
the reported label order. Missing shape parameters default to 0.5, local/facial
parameters to zero, and omitted pose transforms to identity. Unknown controls,
invalid shapes, inconsistent batches and malformed model data return errors.

## C / C++ and C#

The public contract is [include/anny.h](include/anny.h), ABI version 1. Build:

```sh
cargo build --release -p anny-capi --locked
mkdir -p output
cc -Wall -Wextra -Werror -I include examples/c_smoke.c \
  -L target/release -lanny_capi -Wl,-rpath,"$PWD/target/release" -o output/c_smoke
./output/c_smoke data
```

On Linux the library is `target/release/libanny_capi.so`; the corresponding
platform build produces a DLL or dylib. Do not expose Rust-native structures over
the language boundary. The C API uses opaque model/output handles, caller-visible
status codes, thread-local error messages and read-only borrowed tensor views.
Tensor views remain valid only while their owning handle is alive. See the
header's ownership and thread-safety notes before embedding.

A .NET 10 example with SafeHandle ownership and managed copies is included:

```sh
cargo run --release -p anny-cli --locked -- prepare \
  --assets data --output output/anny-default.safetensors
LD_LIBRARY_PATH="$PWD/target/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  dotnet run --project examples/csharp -- "$PWD/output/anny-default.safetensors"
```

The C API was exercised from a real C executable. C# is an integration example,
not a complete Unity package; the browser example is not a character-editor UI.

## WebAssembly

The core can load an in-memory prepared model without filesystem access. The
WASM wrapper copies output into owned JavaScript typed arrays. A compile check:

```sh
rustup target add wasm32-unknown-unknown
cargo check -p anny-wasm --target wasm32-unknown-unknown --locked
```

To run the included local-file browser smoke page, build bindings with wasm-pack:

```sh
wasm-pack build crates/anny-wasm --release --target web \
  --out-dir ../../examples/browser/pkg
python -m http.server --directory examples/browser 8000
```

Open `http://localhost:8000`, select a prepared `.safetensors` model and evaluate.
Python in the last command is merely a replaceable static web server. It is not
part of the browser's model runtime. Account for file size, browser memory and
copies; no WebGPU backend is implemented in this PR.

## Measurements, sampling and fitting

```sh
./target/release/anny measure --assets data
./target/release/anny sample --assets data --options examples/sample-options.json \
  --output output/random-params.json
./target/release/anny generate --assets data --params output/random-params.json \
  --obj output/random.obj
./target/release/anny fit --assets data --target output/character.safetensors \
  --options examples/fit-options.json --output output/fitted-params.json
```

Fitting expects corresponding vertices in the selected model's topology, not an
arbitrary unrelated scan. Its implemented baseline uses joint registration and
finite-difference shape fitting; optional upstream `post_gd`/autograd refinement
is **not implemented**. Sampling follows the calibrated distributions but uses a
native RNG, so a PyTorch seed does not produce the same random character.

## Lightweight Python-reference validation

Generate reference results in the original Anny environment (four cases by
default), then consume them with the native binary:

```sh
PYTHONPATH=/home/slurp/Source/Not_Saved/anny/src \
  python tools/export_reference.py --output output/fixtures
python tools/check_parity.py --assets data --fixtures output/fixtures \
  --summary output/parity-summary.json
```

Use `--cases all-cases` on the exporter for all 23 configurations. Reference
creation may build large caches; this is optional, not a prerequisite for using
the library. Expected inputs are serialized explicitly: there is no assumption
of identical cross-language random-number generators. Missing fixtures fail;
nothing is silently counted as a passed parity check.

The strict comparison checks all saved vertex/joint arrays, face and UV
connectivity, packed skinning weights/indices, base vertex mapping and labels.
Integer arrays/labels are exact. Floats use `atol=1e-6`, `rtol=0`. A successful
23-case run is strong regression coverage, not an exhaustive proof for all inputs.

## Add the imported assets to the existing PR

```sh
git switch codex/anny-rust-native-port
git status --short
git add -- data
git diff --cached --stat
git commit -m "Import pinned Anny assets and portable tensor conversions"
git push origin codex/anny-rust-native-port
```

Only `data/` is staged in these commands. Do not `git add .` after generating
hundreds of megabytes of prepared model caches.

## Licensing

The port preserves NAVER's Apache-2.0 attribution. MakeHuman/MPFB2 and Face Units
assets have their own accompanying CC0 notices; SOMA assets have accompanying
Apache-2.0 notices. CPU projection/BVH compatibility code adapts NVIDIA Warp's
Apache-2.0 implementation and is attributed in [NOTICE](NOTICE). SMPL/SMPL-X
assets and correspondence maps are separate, opt-in inputs with separate terms.
No asset conversion changes the original license. See [LICENSE](LICENSE).
