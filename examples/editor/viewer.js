// The viewport: a three.js scene plus the geometry machinery that keeps the rendered mesh in step
// with whatever the model last produced.
//
// The model's vertices are shared-topology (13,718 of them for the reference character, one list
// indexed by faces), while the viewport wants per-corner vertices as soon as a texture is applied,
// because UVs live on face corners. So the render geometry may hold more vertices than the model
// does, and `sourceIndex` maps each render vertex back to the model's own vertex. Positions, normals
// and the interaction all go through that map.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export class Viewer {
  constructor(canvas) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x14161a);
    this.camera = new THREE.PerspectiveCamera(35, 1, 0.1, 100);
    this.camera.position.set(0, 1.1, 2.6);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(0, 0.95, 0);
    this.controls.update();
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x333844, 1.6));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(2, 4, 3);
    this.scene.add(key);

    this.material = new THREE.MeshStandardMaterial({ color: 0xb4b4b4, roughness: 0.6, metalness: 0, side: THREE.DoubleSide });
    this.texture = null;
    this.mesh = null;
    this.sourceIndex = null;
    this.sourceCount = 0;
    this.shared = { positions: null, normals: null, faces: null };
    this.triangles = 0;
    this.vertices = 0;
    this.digest = 0;

    this.grid = new THREE.GridHelper(4, 20, 0x2c313a, 0x22262c);
    this.grid.position.y = -0.001;
    this.scene.add(this.grid);

    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
  }

  resize() {
    const canvas = this.renderer.domElement;
    const width = canvas.clientWidth || 1;
    const height = canvas.clientHeight || 1;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  /**
   * Build the render geometry. `uv` is the model's per-corner UV index arrays when it has them, in
   * which case every face corner becomes its own render vertex and gets the model's UV; without them
   * the render vertices are the model's own and the map is the identity.
   */
  setTopology({ faces, uv }, vertexCount) {
    if (this.mesh) { this.scene.remove(this.mesh); this.mesh.geometry.dispose(); }
    this.sourceCount = vertexCount;
    this.shared.faces = faces;
    this.shared.positions = new Float32Array(vertexCount * 3);
    this.shared.normals = new Float32Array(vertexCount * 3);
    this.triangles = faces.length / 3;

    const geometry = new THREE.BufferGeometry();
    const corners = uv ? faces.length : 0;
    if (corners) {
      const index = new Uint32Array(corners);
      const sourceIndex = new Uint32Array(corners);
      const uvs = new Float32Array(corners * 2);
      for (let corner = 0; corner < corners; corner++) {
        const source = faces[corner];
        index[corner] = corner;
        sourceIndex[corner] = source;
        const uvIndex = uv.cornerIndices[corner];
        uvs[corner * 2] = uv.coordinates[uvIndex * 2];
        uvs[corner * 2 + 1] = uv.coordinates[uvIndex * 2 + 1];
      }
      geometry.setIndex(new THREE.BufferAttribute(index, 1));
      geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
      this.sourceIndex = sourceIndex;
      this.vertices = corners;
    } else {
      geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(faces), 1));
      this.sourceIndex = null;
      this.vertices = vertexCount;
    }
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(this.vertices * 3), 3));
    geometry.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(this.vertices * 3), 3));
    geometry.setDrawRange(0, this.triangles * 3);
    this.mesh = new THREE.Mesh(geometry, this.material);
    this.mesh.frustumCulled = false;
    this.scene.add(this.mesh);
    return this;
  }

  /**
   * Take the model's vertices, recompute smooth normals over the shared topology and scatter both to
   * the render vertices. Accumulating face normals by their cross product is area weighted, and it
   * runs every update: 27k faces is well under a millisecond, and stale normals on a posed character
   * look worse than the cost of not having them.
   */
  update(vertices) {
    const { faces, positions, normals } = this.shared;
    positions.set(vertices.subarray(0, positions.length));
    normals.fill(0);
    for (let i = 0; i < faces.length; i += 3) {
      const a = faces[i] * 3, b = faces[i + 1] * 3, c = faces[i + 2] * 3;
      const abx = positions[b] - positions[a], aby = positions[b + 1] - positions[a + 1], abz = positions[b + 2] - positions[a + 2];
      const acx = positions[c] - positions[a], acy = positions[c + 1] - positions[a + 1], acz = positions[c + 2] - positions[a + 2];
      const nx = aby * acz - abz * acy, ny = abz * acx - abx * acz, nz = abx * acy - aby * acx;
      for (const v of [a, b, c]) { normals[v] += nx; normals[v + 1] += ny; normals[v + 2] += nz; }
    }
    for (let i = 0; i < positions.length; i += 3) {
      const length = Math.hypot(normals[i], normals[i + 1], normals[i + 2]) || 1;
      normals[i] /= length; normals[i + 1] /= length; normals[i + 2] /= length;
    }

    const attributes = this.mesh.geometry.attributes;
    const renderPositions = attributes.position.array;
    const renderNormals = attributes.normal.array;
    const source = this.sourceIndex;
    if (source) {
      for (let v = 0; v < this.vertices; v++) {
        const s = source[v] * 3, r = v * 3;
        renderPositions[r] = positions[s];
        renderPositions[r + 1] = positions[s + 1];
        renderPositions[r + 2] = positions[s + 2];
        renderNormals[r] = normals[s];
        renderNormals[r + 1] = normals[s + 1];
        renderNormals[r + 2] = normals[s + 2];
      }
    } else {
      renderPositions.set(positions);
      renderNormals.set(normals);
    }
    attributes.position.needsUpdate = true;
    attributes.normal.needsUpdate = true;
    this.digest = digestOf(renderPositions);
    return this;
  }

  /** Frame the character: the model is authored in metres with its feet near the origin. */
  frame() {
    const positions = this.mesh.geometry.attributes.position.array;
    const box = new THREE.Box3();
    for (let v = 0; v < positions.length; v += 3) box.expandByPoint(new THREE.Vector3(positions[v], positions[v + 1], positions[v + 2]));
    const size = box.getSize(new THREE.Vector3());
    const centre = box.getCenter(new THREE.Vector3());
    this.controls.target.copy(centre);
    const distance = Math.max(size.x, size.y, size.z) * 1.9 || 2;
    this.camera.position.set(centre.x, centre.y + size.y * 0.12, centre.z + distance);
    this.camera.near = Math.max(distance / 100, 0.01);
    this.camera.far = distance * 20;
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  setTexture(texture) {
    this.texture = texture;
    this.material.map = texture;
    this.material.needsUpdate = true;
  }

  render() { this.renderer.render(this.scene, this.camera); }
}

/** FNV-1a over the rendered positions: a cheap way to see that a change reached the viewport. */
export function digestOf(values) {
  let hash = 0x811c9dc5;
  for (let i = 0; i < values.length; i++) {
    hash ^= Math.round(values[i] * 1e5) | 0;
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}
