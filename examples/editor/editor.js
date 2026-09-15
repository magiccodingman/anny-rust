// Anny Rust editor: a browser front end for the native runtime's WASM bindings.
//
// Everything the viewport shows comes out of the model itself — `describe()` names the phenotype,
// body, face and bone parameters this build actually has, `evaluate` produces the mesh for a set of
// parameters, and the pose session re-poses that mesh without re-evaluating the shape. There is no
// hard-coded character knowledge here, so loading a different prepared model rebuilds the panels.
import * as THREE from 'three';
import { buildParameters, buildPose, boneMatrix, rng, h } from './controls.js';
import { parseClip, sampleClip } from './clip.js';
import { Viewer } from './viewer.js';

const query = new URLSearchParams(location.search);
const MODEL_PATH = query.get('model') || '../../output/ci-model.safetensors';
const WASM_PATH = query.get('wasm') || '../../output/wasm-web/anny_wasm.js';

const logLines = [];
const logEl = document.querySelector('#log');
const busyEl = document.querySelector('#busy');
const statsEl = document.querySelector('#stats');
const $ = id => document.querySelector(id);

function note(message) {
  logLines.push(message);
  logEl.textContent = logLines.slice(-40).join('\n');
}

const state = {
  wasm: null,
  model: null,
  modelF32: null,
  describe: {},
  ready: false,
  status: 'starting',
  error: null,
  parameters: { phenotype_kwargs: {}, local_changes_kwargs: {}, facial_actions: {} },
  pose: {},
  poseClipping: false,
  session: null,
  clip: null,
  clipTime: 0,
  clipPlaying: false,
  batch: 0,
  stats: { updates: 0, lastMs: 0, source: 'none', digest: 0 },
};

const viewer = new Viewer($('#canvas'));

/** Parse whatever the bindings hand back: these fields are JSON values, not always strings. */
function asObject(value) {
  if (typeof value === 'string') return JSON.parse(value);
  return value;
}

// ---------------------------------------------------------------- parameters

/** Positional arguments of the model's own parameter sets, in the shape the API expects. */
function parametersJson() {
  return {
    phenotype_kwargs: state.parameters.phenotype_kwargs,
    local_changes_kwargs: state.parameters.local_changes_kwargs,
    facial_actions: state.parameters.facial_actions,
    pose_parameters: poseJson(),
    pose_parameterization: 'local-bone',
  };
}

/** Only bones the user has moved, as the row-major 4x4 matrices the pose API takes. */
function poseJson() {
  const out = {};
  for (const [bone, rotation] of Object.entries(state.pose)) {
    if (rotation.x || rotation.y || rotation.z) out[bone] = boneMatrix(rotation);
  }
  return out;
}

let shapeTimer = null;
function applyShape({ immediate = false } = {}) {
  if (!state.modelF32) return;
  if (!immediate) {
    clearTimeout(shapeTimer);
    shapeTimer = setTimeout(() => applyShape({ immediate: true }), 60);
    return;
  }
  const started = performance.now();
  const result = state.modelF32.evaluate(JSON.stringify(parametersJson()));
  // A session is pinned to the shape it was built from, so every shape change makes a new one. Pose
  // changes then ride this session instead of re-evaluating: that is the 1.5-2x measured on the
  // native side, and it is the difference between interactive posing and a re-evaluation per frame.
  state.session = state.modelF32.pose_session(JSON.stringify(parametersJson()));
  pushVertices(result.tensor('vertices'), 'evaluate');
  state.stats.lastMs = performance.now() - started;
  state.poseClipping = false;
}

function applyPose() {
  if (!state.session) { applyShape({ immediate: true }); return; }
  const started = performance.now();
  state.session.update(JSON.stringify(poseJson()));
  pushVertices(state.session.tensor('vertices'), 'session');
  state.stats.lastMs = performance.now() - started;
}

function pushVertices(vertices, source) {
  viewer.update(vertices);
  state.stats.updates += 1;
  state.stats.source = source;
  state.stats.digest = viewer.digest;
}

// ---------------------------------------------------------------- panels

function buildPanels() {
  const describe = state.describe;
  const phenotype = describe.phenotype_labels || [];
  const body = describe.local_change_labels || [];
  const face = describe.facial_action_labels || [];

  buildParameters($('#phenotype-controls'), phenotype, name => state.parameters.phenotype_kwargs[name] ?? 0,
    (name, value) => { state.parameters.phenotype_kwargs[name] = value; }, () => applyShape());
  buildParameters($('#body-controls'), body, name => state.parameters.local_changes_kwargs[name] ?? 0,
    (name, value) => { state.parameters.local_changes_kwargs[name] = value; }, () => applyShape());
  buildParameters($('#face-controls'), face, name => state.parameters.facial_actions[name] ?? 0,
    (name, value) => { state.parameters.facial_actions[name] = value; }, () => applyShape());

  const pose = describe.bone_labels || [];
  const parents = describe.bone_parents || pose.map(() => null);
  buildPose($('#pose-controls'), { labels: pose, parents }, state.pose, () => { state.poseClipping = false; applyPose(); }, $('#bone-filter').value);

  $('#section-body').open = body.length > 0;
  $('#section-face').open = face.length > 0;
  $('#section-pose').open = pose.length > 0;
  $('#model-summary').textContent = [
    `${describe.vertices ?? viewer.vertices} vertices`,
    `${describe.faces ?? viewer.triangles} faces`,
    `${pose.length} bones`,
    `${phenotype.length + body.length + face.length} blendshapes`,
  ].join(' · ');
}

function buildPresets() {
  const list = $('#preset-list');
  list.replaceChildren();
  const presets = {
    neutral: {},
    tall: { phenotype: [['height', 1]] },
    short: { phenotype: [['height', 0.2]] },
    heavy: { phenotype: [['weight', 1]] },
    light: { phenotype: [['weight', 0.1]] },
    smile: { face: [['smile', 1]] },
    frown: { face: [['smile', 0]] },
  };
  for (const [name, spec] of Object.entries(presets)) {
    list.append(h('button', { text: name, title: JSON.stringify(spec), onclick: () => applyPreset(spec) }));
  }
  for (const [name, saved] of savedPresets()) {
    list.append(h('button', { text: name, title: 'Saved preset', onclick: () => loadStateObject(saved) }));
  }
}

/** Preset fields name parameter labels the way a person would ("height"); a label that this model
 *  does not expose is skipped rather than invented. */
function applyPreset(spec) {
  for (const [group, assignments] of Object.entries(spec)) {
    const labels = group === 'phenotype' ? state.describe.phenotype_labels || []
      : group === 'face' ? state.describe.facial_action_labels || []
      : state.describe.local_change_labels || [];
    const target = group === 'phenotype' ? state.parameters.phenotype_kwargs
      : group === 'face' ? state.parameters.facial_actions
      : state.parameters.local_changes_kwargs;
    for (const [needle, value] of assignments) {
      for (const label of labels) if (label.toLowerCase().includes(needle)) target[label] = value;
    }
  }
  refreshControls();
  applyShape({ immediate: true });
}

function randomise(seed = Number($('#random-seed').value) || 1) {
  const random = rng(seed);
  for (const [labels, target] of [
    [state.describe.phenotype_labels || [], state.parameters.phenotype_kwargs],
    [state.describe.local_change_labels || [], state.parameters.local_changes_kwargs],
    [state.describe.facial_action_labels || [], state.parameters.facial_actions],
  ]) for (const label of labels) target[label] = Math.round(random() * 100) / 100;
  refreshControls();
  applyShape({ immediate: true });
}

function resetParameters() {
  state.parameters = { phenotype_kwargs: {}, local_changes_kwargs: {}, facial_actions: {} };
  state.pose = {};
  state.poseClipping = false;
  refreshControls();
  buildPanels();
  applyShape({ immediate: true });
}

/** Put the DOM back in step with the state after a preset, a randomise or a state load. */
function refreshControls() {
  for (const [id, values] of [
    ['#phenotype-controls', state.parameters.phenotype_kwargs],
    ['#body-controls', state.parameters.local_changes_kwargs],
    ['#face-controls', state.parameters.facial_actions],
  ]) {
    for (const row of $(id).children) if (row.dataset.name in values) row.set(values[row.dataset.name]);
  }
  setMaterialInputs();
}

// ---------------------------------------------------------------- state files

const STATE_VERSION = 1;
const material = () => viewer.material;

function stateObject() {
  return {
    version: STATE_VERSION,
    parameters: state.parameters,
    pose: state.pose,
    material: {
      colour: `#${material().color.getHexString()}`,
      roughness: material().roughness,
      metalness: material().metalness,
      wireframe: material().wireframe,
      texture: viewer.texture ? viewer.texture.name || null : null,
    },
    camera: { position: viewer.camera.position.toArray(), target: viewer.controls.target.toArray() },
    batch: state.batch,
  };
}

function applyStateObject(saved) {
  state.parameters = {
    phenotype_kwargs: saved.parameters?.phenotype_kwargs || {},
    local_changes_kwargs: saved.parameters?.local_changes_kwargs || {},
    facial_actions: saved.parameters?.facial_actions || {},
  };
  state.pose = saved.pose || {};
  if (saved.material) {
    const colour = saved.material.colour || '#b4b4b4';
    material().color = new THREE.Color(colour);
    material().roughness = saved.material.roughness ?? 0.6;
    material().metalness = saved.material.metalness ?? 0;
    material().wireframe = Boolean(saved.material.wireframe);
  }
  if (saved.camera?.position) viewer.camera.position.fromArray(saved.camera.position);
  if (saved.camera?.target) viewer.controls.target.fromArray(saved.camera.target);
  viewer.controls.update();
  state.batch = saved.batch ?? 0;
  $('#batch-index').value = String(state.batch);
  buildPanels();
  setMaterialInputs();
}

function savedPresets() {
  try { return Object.entries(JSON.parse(localStorage.getItem('anny-presets') || '{}')); } catch { return []; }
}

function setMaterialInputs() {
  $('#material-colour').value = `#${material().color.getHexString()}`;
  $('#material-roughness').value = String(material().roughness);
  $('#material-metalness').value = String(material().metalness);
  $('#material-wireframe').checked = Boolean(material().wireframe);
}

function download(name, bytes, type = 'application/octet-stream') {
  const url = URL.createObjectURL(new Blob([bytes], { type }));
  const link = h('a', { href: url, download: name });
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

// ---------------------------------------------------------------- model loading

async function loadModel(urlOrBytes) {
  state.ready = false;
  state.status = 'loading';
  busyEl.classList.add('on');
  try {
    let bytes;
    if (typeof urlOrBytes === 'string') {
      const response = await fetch(urlOrBytes);
      if (!response.ok) throw new Error(`${urlOrBytes}: HTTP ${response.status}`);
      const total = Number(response.headers.get('content-length')) || 0;
      bytes = new Uint8Array(await response.arrayBuffer());
      note(`Loaded ${(bytes.length / 1e6).toFixed(1)} MB of ${total ? (total / 1e6).toFixed(1) + ' MB' : 'model payload'}`);
    } else {
      bytes = urlOrBytes;
    }
    state.model = new state.wasm.AnnyModel(bytes);
    state.modelF32 = state.model.to_f32();
    state.describe = asObject(state.modelF32.describe());
    state.session = null;
    state.clip = null;
    state.clipTime = 0;

    // `shape` lives on a result, not on the f32 model, so the vertex count comes from one evaluate —
    // which is also the mesh the viewport shows before any parameter is touched.
    const rest = state.modelF32.evaluate(JSON.stringify({}));
    const restShape = rest.shape('vertices');
    const vertexCount = restShape[restShape.length - 2];
    viewer.setTopology({ faces: Int32Array.from(state.modelF32.indices('faces')), uv: textureCoordinates() }, vertexCount);
    buildPanels();
    buildPresets();
    setMaterialInputs();
    applyShape({ immediate: true });
    viewer.frame();
    state.ready = true;
    state.status = 'ready';
    note(`Ready: ${viewer.vertices} render vertices, ${viewer.triangles} triangles`);
  } catch (error) {
    state.error = String(error && error.stack ? error.stack : error);
    state.status = 'failed';
    note(`Failed: ${state.error}`);
    throw error;
  } finally {
    busyEl.classList.remove('on');
  }
}

/** The model's UVs live on face corners: `texture_coordinates` is the list, the face array indexes
 *  into it. Without both, the viewport simply has no UV attribute. */
function textureCoordinates() {
  try {
    const coordinates = Float32Array.from(state.modelF32.tensor('texture_coordinates') ?? []);
    const cornerIndices = Int32Array.from(state.modelF32.indices('face_texture_coordinate_indices') ?? []);
    if (!coordinates.length || !cornerIndices.length) return null;
    return { coordinates, cornerIndices };
  } catch (error) {
    note(`No texture coordinates: ${error}`);
    return null;
  }
}

// ---------------------------------------------------------------- clips

function loadClip(json) {
  state.clip = parseClip(json);
  state.clipTime = 0;
  const time = $('#clip-time');
  time.max = String(state.clip.duration);
  time.value = '0';
  $('#clip-summary').textContent = `${state.clip.name}: ${state.clip.frames.length} frames, ${state.clip.duration.toFixed(2)}s`;
  return state.clip;
}

function seekClip(time) {
  if (!state.clip || !state.session) return;
  state.clipTime = time;
  state.poseClipping = true;
  state.session.update(JSON.stringify(sampleClip(state.clip, time)));
  pushVertices(state.session.tensor('vertices'), 'clip');
}

// ---------------------------------------------------------------- export

function exportGlb() {
  const options = {
    name: 'Anny',
    translation: [0, 0, 0],
    color: [material().color.r, material().color.g, material().color.b, 1],
    rigged: true,
    batch_index: state.batch,
  };
  // The GLB writer lives on the f64 model: the f32 mirror has no exporter.
  const bytes = state.model.export_glb(JSON.stringify(parametersJson()), JSON.stringify(options));
  window.__anny_editor.lastGlb = bytes;
  note(`Exported GLB: ${(bytes.length / 1e6).toFixed(2)} MB`);
  return bytes;
}

// ---------------------------------------------------------------- wiring

function wire() {
  $('#model-path').value = MODEL_PATH;
  $('#load-model').addEventListener('click', () => { loadModel($('#model-path').value).catch(() => {}); });
  $('#model-file').addEventListener('change', async event => {
    const file = event.target.files[0];
    if (file) loadModel(new Uint8Array(await file.arrayBuffer())).catch(() => {});
  });
  $('#export-glb').addEventListener('click', () => {
    const bytes = exportGlb();
    download('anny.glb', bytes, 'model/gltf-binary');
  });
  $('#save-state').addEventListener('click', () => {
    download('anny-state.json', JSON.stringify(stateObject(), null, 2), 'application/json');
  });
  $('#load-state').addEventListener('click', () => $('#state-file').click());
  $('#state-file').addEventListener('change', async event => {
    const file = event.target.files[0];
    if (!file) return;
    const saved = JSON.parse(await file.text());
    applyStateObject(saved);
    applyShape({ immediate: true });
  });
  $('#batch-index').addEventListener('change', event => { state.batch = Number(event.target.value) || 0; });

  $('#material-colour').addEventListener('input', event => { material().color = new THREE.Color(event.target.value); });
  $('#material-roughness').addEventListener('input', event => { material().roughness = Number(event.target.value); });
  $('#material-metalness').addEventListener('input', event => { material().metalness = Number(event.target.value); });
  $('#material-wireframe').addEventListener('change', event => { material().wireframe = event.target.checked; });
  $('#material-texture').addEventListener('change', async event => {
    const file = event.target.files[0];
    if (file) await setTexture(await file.arrayBuffer(), file.type, file.name);
  });

  $('#bone-filter').addEventListener('input', () => buildPanels());
  $('#randomise').addEventListener('click', () => randomise());
  $('#reset-parameters').addEventListener('click', resetParameters);
  $('#save-preset').addEventListener('click', () => {
    const presets = JSON.parse(localStorage.getItem('anny-presets') || '{}');
    presets[`saved ${Object.keys(presets).length + 1}`] = stateObject();
    localStorage.setItem('anny-presets', JSON.stringify(presets));
    buildPresets();
  });

  $('#clip-file').addEventListener('change', async event => {
    const file = event.target.files[0];
    if (file) loadClip(await file.text());
  });
  $('#clip-play').addEventListener('click', event => {
    state.clipPlaying = !state.clipPlaying;
    event.target.textContent = state.clipPlaying ? 'Pause' : 'Play';
  });
  $('#clip-reset').addEventListener('click', () => seekClip(0));
  $('#clip-time').addEventListener('input', event => {
    state.clipPlaying = false;
    $('#clip-play').textContent = 'Play';
    seekClip(Number(event.target.value));
  });
}

async function setTexture(bytes, type = 'image/png', name = 'texture.png') {
  const bitmap = await createImageBitmap(new Blob([bytes], { type }));
  const texture = new THREE.Texture(bitmap);
  texture.name = name;
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.flipY = false;
  texture.needsUpdate = true;
  viewer.setTexture(texture);
  $('#material-note').textContent = `${name} (${bitmap.width}×${bitmap.height})`;
  return true;
}

// ---------------------------------------------------------------- animation loop

let previous = performance.now();
function tick() {
  const now = performance.now();
  const delta = (now - previous) / 1000;
  previous = now;
  if (state.clipPlaying && state.clip) {
    const speed = Number($('#clip-speed').value) || 1;
    let time = state.clipTime + delta * speed;
    if (time > state.clip.duration) time = 0;
    seekClip(time);
    $('#clip-time').value = String(time);
  }
  viewer.render();
  statsEl.textContent = state.ready
    ? `${viewer.vertices} vertices · ${viewer.triangles} triangles · ${state.stats.updates} updates · `
      + `last ${state.stats.lastMs.toFixed(1)} ms via ${state.stats.source}`
    : state.status;
  requestAnimationFrame(tick);
}

// ---------------------------------------------------------------- automation surface

// The qualification harness drives the editor through this object, and through the real DOM controls
// as well: the point is to prove the page works, not that a second API does.
window.__anny_editor = {
  get ready() { return state.ready; },
  get status() { return state.status; },
  get error() { return state.error; },
  get log() { return logLines.slice(); },
  get lastGlb() { return this._glb || null; },
  set lastGlb(value) { this._glb = value; },
  describe: () => state.describe,
  parameters: () => JSON.parse(JSON.stringify(state.parameters)),
  pose: () => JSON.parse(JSON.stringify(state.pose)),
  stats: () => ({ ...state.stats, vertices: viewer.vertices, triangles: viewer.triangles }),
  digest: () => viewer.digest,
  setParameter: (group, name, value) => {
    const target = group === 'phenotype' ? state.parameters.phenotype_kwargs
      : group === 'face' ? state.parameters.facial_actions : state.parameters.local_changes_kwargs;
    target[name] = value;
    applyShape({ immediate: true });
    return viewer.digest;
  },
  setPose: (bone, degrees) => {
    state.pose[bone] = { x: degrees.x || 0, y: degrees.y || 0, z: degrees.z || 0 };
    state.poseClipping = false;
    applyPose();
    return viewer.digest;
  },
  randomise: seed => { randomise(seed); return viewer.digest; },
  exportGlb: () => { exportGlb(); return { bytes: window.__anny_editor.lastGlb.length, digest: viewer.digest }; },
  saveState: () => JSON.stringify(stateObject()),
  loadState: json => {
    applyStateObject(typeof json === 'string' ? JSON.parse(json) : json);
    applyShape({ immediate: true });
    return viewer.digest;
  },
  setTexture: (bytes, name) => setTexture(bytes, name?.endsWith?.('.jpg') ? 'image/jpeg' : 'image/png', name || 'texture.png'),
  loadClip: json => {
    const clip = loadClip(json);
    return { name: clip.name, frames: clip.frames.length, duration: clip.duration };
  },
  seekClip: time => { seekClip(time); return viewer.digest; },
  setMaterial: values => {
    if (values.colour) material().color = new THREE.Color(values.colour);
    if (values.roughness !== undefined) material().roughness = values.roughness;
    if (values.metalness !== undefined) material().metalness = values.metalness;
    if (values.wireframe !== undefined) material().wireframe = values.wireframe;
    setMaterialInputs();
    return true;
  },
  camera: () => ({ position: viewer.camera.position.toArray(), target: viewer.controls.target.toArray() }),
  texture: () => (viewer.texture ? viewer.texture.name : null),
  /** What the last frame actually drew: a viewport that renders nothing still has a live page. */
  renderInfo: () => ({ triangles: viewer.renderer.info.render.triangles, calls: viewer.renderer.info.render.calls }),
  rebase: () => { viewer.frame(); viewer.render(); return viewer.digest; },
};

// ---------------------------------------------------------------- boot

(async () => {
  wire();
  requestAnimationFrame(tick);
  try {
    state.wasm = await import(WASM_PATH);
    await state.wasm.default();
    note(`WASM ready: ${WASM_PATH}`);
    await loadModel(MODEL_PATH);
  } catch (error) {
    state.error = String(error && error.stack ? error.stack : error);
    state.status = 'failed';
    note(`Boot failed: ${state.error}`);
  }
})();
