#!/usr/bin/env node
// Runs the WebAssembly bindings under Node, so the browser-facing classes are *executed* rather than
// only compiled: the pose session in both precisions, the ownership rule that lets the model object be
// dropped while a session lives, and the f64 -> f32 conversion the browser and Unity paths use.
//
// Usage: node wasm-node-smoke.cjs <wasm-bindgen-out-dir> <model.safetensors>
"use strict";

const fs = require("node:fs");
const path = require("node:path");

const [bindgenDir, modelPath] = process.argv.slice(2);
if (!bindgenDir || !modelPath) {
  console.error("usage: wasm-node-smoke.cjs <bindgen-out-dir> <model.safetensors>");
  process.exit(2);
}

const { AnnyModel } = require(path.resolve(bindgenDir, "anny_wasm.js"));

// The same pose the C and C# smoke tests use: a 20 degree rotation with a translation.
const pose = {
  root: [
    [1, 0, 0, 0.2],
    [0, 0.9396926, 0.3420201, -0.15],
    [0, -0.3420201, 0.9396926, 0.05],
    [0, 0, 0, 1],
  ],
};
const parameters = JSON.stringify({ pose_parameters: pose });
const poseJson = JSON.stringify(pose);

function fail(message) {
  console.error(`FAIL: ${message}`);
  process.exit(1);
}

function require_(condition, message) {
  if (!condition) fail(message);
}

/** Bit-for-bit comparison: the session must not merely be close to `evaluate`. */
function identical(left, right) {
  if (left.length !== right.length) return false;
  for (let i = 0; i < left.length; i += 1) if (left[i] !== right[i]) return false;
  return true;
}

const bytes = new Uint8Array(fs.readFileSync(path.resolve(modelPath)));

// --- f64 -----------------------------------------------------------------------------------------
const model = new AnnyModel(bytes);
const evaluated = model.evaluate(parameters);
const session = model.pose_session(parameters);
session.update(poseJson);

const evaluatedVertices = evaluated.tensor("vertices");
const sessionVertices = session.tensor("vertices");
require_(sessionVertices.length > 0, "f64 session returned no vertices");
require_(
  identical(sessionVertices, evaluatedVertices),
  "f64 session output differs from evaluate",
);
require_(session.coefficients().length > 0, "f64 session returned no coefficients");

// A session holds its own reference to the model: freeing the model object must not disturb it.
const f64Snapshot = Float64Array.from(sessionVertices);
model.free();
session.update(poseJson);
require_(
  identical(session.tensor("vertices"), f64Snapshot),
  "f64 session changed after its model was freed",
);

// --- f32, through the conversion the browser and Unity paths use ---------------------------------
const f64Model = new AnnyModel(bytes);
const single = f64Model.to_f32();
const singleEvaluated = single.evaluate(parameters);
const singleSession = single.pose_session(parameters);
singleSession.update(poseJson);
const singleVertices = singleSession.tensor("vertices");

require_(singleVertices.length > 0, "f32 session returned no vertices");
require_(
  identical(singleVertices, singleEvaluated.tensor("vertices")),
  "f32 session output differs from evaluate",
);
require_(singleSession.coefficients().length > 0, "f32 session returned no coefficients");

const f32Snapshot = Float32Array.from(singleVertices);
single.free();
singleSession.update(poseJson);
require_(
  identical(singleSession.tensor("vertices"), f32Snapshot),
  "f32 session changed after its model was freed",
);

console.log(
  `WASM under Node: ${evaluatedVertices.length / 3} vertices; f64 and f32 sessions match evaluate ` +
    "exactly and survive their model being freed",
);
