// Panel construction for the Anny Rust editor.
//
// Everything here is a plain function over DOM nodes: the editor owns the state and passes
// callbacks, so the panels can be rebuilt whenever a model with a different parameter set is loaded.

/** `document.createElement` with attributes, children and an optional click handler. */
export function h(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === 'class') node.className = value;
    else if (key === 'text') node.textContent = value;
    else if (key === 'oninput') node.addEventListener('input', value);
    else if (key === 'onchange') node.addEventListener('change', value);
    else if (key === 'onclick') node.addEventListener('click', value);
    else if (value !== undefined && value !== null) node.setAttribute(key, value);
  }
  for (const child of [].concat(children)) if (child) node.append(child);
  return node;
}

/** One labelled range with a numeric readout, the shape every scalar control takes. */
export function slider(name, value, { min = 0, max = 1, step = 0.01, format = v => v.toFixed(2) } = {}) {
  const input = h('input', { type: 'range', min, max, step, value: String(value) });
  const output = h('output', { text: format(value) });
  input.addEventListener('input', () => { output.textContent = format(Number(input.value)); });
  const row = h('div', { class: 'control' }, [h('span', { text: name, title: name }), input, output]);
  row.dataset.name = name;
  row.value = () => Number(input.value);
  row.set = v => {
    input.value = String(v);
    output.textContent = format(v);
  };
  return row;
}

/**
 * A labelled control generated from one label of the model's parameter set.
 *
 * Ranges are a UI convention, not something the model reports: the labels come from `describe()`,
 * which names the blendshapes but not their bounds, so every control defaults to 0..1 and the value
 * is passed through untouched.
 */
function controlFor(label, get, set, onChange) {
  const row = slider(label, get(label));
  row.querySelector('input').addEventListener('input', () => {
    set(label, row.value());
    onChange();
  });
  return row;
}

export function buildParameters(container, labels, get, set, onChange) {
  container.replaceChildren();
  if (!labels.length) {
    container.append(h('div', { class: 'note', text: 'This model exposes no labels in this group.' }));
    return;
  }
  for (const label of labels) container.append(controlFor(label, get, set, onChange));
}

/** Collapsible bones in hierarchy order, each with an X/Y/Z rotation in degrees. */
export function buildPose(container, { labels, parents }, pose, onChange, filter = '') {
  container.replaceChildren();
  const children = new Map();
  labels.forEach((label, i) => {
    const parent = parents[i];
    if (!children.has(parent)) children.set(parent, []);
    children.get(parent).push(label);
  });
  const needle = filter.trim().toLowerCase();
  const shown = needle ? labels.filter(l => l.toLowerCase().includes(needle)) : null;
  const roots = labels.filter((label, i) => !labels.includes(parents[i]));

  const addBone = (parent, label, depth) => {
    if (shown && !shown.includes(label)) return;
    const rotation = pose[label] || { x: 0, y: 0, z: 0 };
    const rows = ['x', 'y', 'z'].map(axis => {
      const row = slider(axis.toUpperCase(), rotation[axis], { min: -180, max: 180, step: 1, format: v => `${Math.round(v)}°` });
      row.querySelector('input').addEventListener('input', () => {
        rotation[axis] = row.value();
        pose[label] = rotation;
        onChange();
      });
      return row;
    });
    const box = h('details', { class: 'bone' }, [
      h('summary', { text: `${'· '.repeat(depth)}${label}`, title: label }),
      ...rows,
    ]);
    container.append(box);
    for (const child of children.get(label) || []) addBone(label, child, depth + 1);
  };
  for (const root of roots) addBone(null, root, 0);
  if (!container.childElementCount) {
    container.append(h('div', { class: 'note', text: needle ? 'No bone matches that filter.' : 'This model exposes no bones.' }));
  }
}

/** The rotation sliders' value for one bone, as the 4x4 nested array the pose API takes. */
export function boneMatrix(rotation) {
  const rad = Math.PI / 180;
  const [x, y, z] = ['x', 'y', 'z'].map(axis => (rotation?.[axis] || 0) * rad);
  const cx = Math.cos(x), sx = Math.sin(x);
  const cy = Math.cos(y), sy = Math.sin(y);
  const cz = Math.cos(z), sz = Math.sin(z);
  // R = Rz * Ry * Rx, written out row by row; the API's matrices are row-major.
  const r = [
    [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
    [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
    [-sy, cy * sx, cy * cx],
  ];
  return [
    [r[0][0], r[0][1], r[0][2], 0],
    [r[1][0], r[1][1], r[1][2], 0],
    [r[2][0], r[2][1], r[2][2], 0],
    [0, 0, 0, 1],
  ];
}

/** `mulberry32`: a seeded generator, so "randomise" is reproducible for a given seed. */
export function rng(seed) {
  let a = (seed >>> 0) || 1;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
