# Native upstream asset import

The CLI can now read the pinned upstream `src/anny/data/` tree directly, including
its original `.pth`, `.pt`, `.yaml`, and `.yml` files. Neither generation nor asset
conversion invokes Python, PyTorch, LibTorch, or Warp. Existing converted assets
remain supported unchanged and are preferred when a portable sibling exists.

## Normal use: no import step is needed

The PR already contains the data payload. From a fresh clone:

```sh
cargo build --workspace --release --locked
./target/release/anny verify-assets --assets data
./target/release/anny generate --assets data --config examples/all.json \
  --params examples/character.json --obj output/character.obj
```

## Import an untouched upstream checkout without Python

```sh
./target/release/anny import-upstream \
  --source /home/slurp/Source/Not_Saved/anny \
  --destination output/native-data
./target/release/anny verify-assets --assets output/native-data
```

The destination must be **new**. Import is staged beside it, every conversion and
hash is verified, then the directory is renamed into place. Existing imported
assets and the original checkout are never overwritten. A failed import cleans
up its staging directory. Symlinks and non-regular source files are rejected.
The pinned revision is checked with Git, including local asset modifications.
An unpacked source archive or a deliberate different-revision experiment requires
`--allow-revision-mismatch true`; its recorded source revision must not be mistaken
for verified pinned provenance. Git is used only to inspect source provenance.

Direct generation does not require even this preparation step:

```sh
./target/release/anny generate \
  --assets /home/slurp/Source/Not_Saved/anny/src/anny/data \
  --config examples/all.json --params examples/character.json \
  --obj output/direct-from-upstream.obj
```

## Supported tensor archive boundary

`torch_archive` is a read-only decoder for the tensor/dictionary subset actually
used by the pinned Anny assets. It recognizes a strict allowlist of pickle data
construction tags; it does not import Python modules, evaluate code, call arbitrary
constructors, or extract archive paths onto disk. Unknown classes/opcodes fail.
ZIP expansion, dimensions, strides, tensor bytes, memo entries, recursion, and
expanded metadata are bounded. Dense tensor views are materialized respecting
storage offsets, strides and endianness. Scalar sparse CSR/COO data is densified;
duplicate sparse indices are rejected rather than overwritten.

The portable output uses the existing Anny Safetensors envelope and preserves
source dtype and nested metadata. This is not a general-purpose PyTorch checkpoint
loader: custom Python objects, arbitrary tensor subclasses, legacy non-ZIP saves,
and unrecognized serialization constructs are not accepted.

## Validation

The native integration test compares **all eight committed tensor archives** to
the already-committed reference conversions: raw tensor bytes, dtypes, shapes,
tensor names, nested metadata, and source SHA-256. It uses no Python:

```sh
cargo test -p anny-core --release --locked --test native_import -- --ignored --nocapture
```

All eight passed locally. Full generation from the untouched upstream data tree
also completed (13,718 vertices, 104 bones). A native import and subsequent
manifest verification completed with 1,568 entries. The offline reference checkout
used for that last smoke lacked `.git`, so its manifest honestly records `unknown`
and the explicit revision-mismatch flag was used. These checks do not assert that
every possible PyTorch archive or every model configuration is supported.

The former `tools/import_assets.py` remains an optional reference/conversion tool;
it is no longer required for the native workflow.
