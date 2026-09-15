// Animation clips: the `frames` JSON the CLI's `motion` command reads, sampled into a pose.
//
// A clip stores, per frame, the same `pose_parameters` the runtime takes: bone name to a row-major
// 4x4 matrix with the translation in the last column. Playback interpolates between the surrounding
// frames, slerping rotations and lerping translations, which is what makes a 3-frame demo loop look
// continuous instead of like a slideshow.
import * as THREE from 'three';

/** Row-major nested matrix, as the API writes them, to a three.js matrix (which is column-major). */
function toMatrix(nested) {
  const flat = [];
  for (let column = 0; column < 4; column++) for (let row = 0; row < 4; row++) flat.push(nested[row][column]);
  return new THREE.Matrix4().fromArray(flat);
}

/** Back to the row-major nested form. */
function fromMatrix(matrix) {
  const e = matrix.elements;
  const out = [];
  for (let row = 0; row < 4; row++) {
    const values = [];
    for (let column = 0; column < 4; column++) values.push(e[column * 4 + row]);
    out.push(values);
  }
  return out;
}

export function parseClip(json) {
  const data = typeof json === 'string' ? JSON.parse(json) : json;
  const frames = (data.frames || [])
    .map(frame => ({ time: Number(frame.time) || 0, pose: frame.pose_parameters || {} }))
    .sort((a, b) => a.time - b.time);
  return {
    name: data.name || 'clip',
    parameters: data.parameters || {},
    frames,
    duration: frames.length ? frames[frames.length - 1].time : 0,
  };
}

/** The pose at `time`, between the two frames that surround it. */
export function sampleClip(clip, time) {
  if (!clip.frames.length) return {};
  let index = 0;
  while (index < clip.frames.length - 1 && clip.frames[index + 1].time <= time) index += 1;
  const a = clip.frames[index];
  const b = clip.frames[Math.min(index + 1, clip.frames.length - 1)];
  const span = b.time - a.time;
  const t = span > 1e-9 ? Math.min(Math.max((time - a.time) / span, 0), 1) : 0;

  const out = {};
  for (const bone of new Set([...Object.keys(a.pose), ...Object.keys(b.pose)])) {
    const from = a.pose[bone] ? toMatrix(a.pose[bone]) : new THREE.Matrix4();
    const to = b.pose[bone] ? toMatrix(b.pose[bone]) : new THREE.Matrix4();
    const rotation = new THREE.Quaternion().setFromRotationMatrix(from)
      .slerp(new THREE.Quaternion().setFromRotationMatrix(to), t);
    const position = new THREE.Vector3().setFromMatrixPosition(from)
      .lerp(new THREE.Vector3().setFromMatrixPosition(to), t);
    out[bone] = fromMatrix(new THREE.Matrix4().compose(position, rotation, new THREE.Vector3(1, 1, 1)));
  }
  return out;
}
