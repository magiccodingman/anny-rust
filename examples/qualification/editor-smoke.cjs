// Optional development qualification: drives the browser editor in a real browser.
//
// The editor is a user interface, so this checks user-facing behaviour end to end: panels built from
// the model's own `describe()`, sliders that move the mesh, posing through the pose session, GLB
// export validated by the official glTF validator, state save/load, a texture, and clip playback.
// Results are written after every check so a hang still leaves evidence behind.
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const zlib = require('node:zlib');
const { chromium } = require('playwright');
const { validateBytes } = require('gltf-validator');

const ROOT = path.resolve(__dirname, '../..');
const RESULTS = path.join(ROOT, 'output/editor-result.json');
const EXPORT = path.join(ROOT, 'output/editor-export.glb');
const MIME = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.wasm': 'application/wasm',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.safetensors': 'application/octet-stream',
};

const results = { checks: {}, failedRequests: [], httpErrors: [], console: [], pageErrors: [], ok: false };
let server;
let browser;

function save() {
  fs.mkdirSync(path.dirname(RESULTS), { recursive: true });
  fs.writeFileSync(RESULTS, JSON.stringify(results, null, 2));
}

function record(name, check) {
  results.checks[name] = check;
  save();
  const flag = check.ok ? 'ok  ' : 'FAIL';
  console.log(`${flag} ${name}: ${JSON.stringify(check)}`);
}

function serve() {
  return new Promise((resolve) => {
    server = http.createServer((request, response) => {
      const url = new URL(request.url, 'http://127.0.0.1');
      if (url.pathname === '/favicon.ico') {
        response.writeHead(204).end();
        return;
      }
      const file = path.join(ROOT, decodeURIComponent(url.pathname));
      if (!file.startsWith(ROOT) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
        if (url.pathname !== '/favicon.ico') {
          results.httpErrors.push({ path: url.pathname, status: 404 });
          save();
        }
        response.writeHead(404).end('not found');
        return;
      }
      const headers = { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' };
      if (path.extname(file) === '.safetensors') headers['Content-Length'] = fs.statSync(file).size;
      response.writeHead(200, headers);
      fs.createReadStream(file).pipe(response);
    });
    server.listen(0, '127.0.0.1', () => resolve(`http://127.0.0.1:${server.address().port}/`));
  });
}

/** A 2x2 red/blue PNG, so the texture check needs no fixture on disk. */
function tinyPng() {
  const chunk = (type, data) => {
    const body = Buffer.concat([Buffer.from(type, 'ascii'), data]);
    const crc = Buffer.alloc(4);
    crc.writeUInt32BE(zlib.crc32(body) >>> 0);
    const length = Buffer.alloc(4);
    length.writeUInt32BE(data.length);
    return Buffer.concat([length, body, crc]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(2, 0);
  ihdr.writeUInt32BE(2, 4);
  ihdr[8] = 8;
  ihdr[9] = 2;
  const raw = Buffer.from([0, 255, 0, 0, 0, 0, 255, 1, 0, 0, 255, 255, 0, 0]);
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', zlib.deflateSync(raw)),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

async function main() {
  const address = await serve();
  browser = await chromium.launch({
    headless: true,
    ...(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {}),
  });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  page.on('pageerror', error => {
    results.pageErrors.push(String(error));
    save();
  });
  page.on('console', message => {
    if (message.type() === 'error') {
      results.console.push(message.text());
      save();
    }
  });
  page.on('requestfailed', request => {
    results.failedRequests.push({ url: request.url(), error: request.failure()?.errorText });
    save();
  });

  const textureFile = path.join(ROOT, 'output/editor-texture.png');
  fs.writeFileSync(textureFile, tinyPng());

  console.log(`editor: ${address}examples/editor/index.html`);
  await page.goto(`${address}examples/editor/index.html`, { waitUntil: 'domcontentloaded' });

  const editor = (expression, argument) => page.evaluate(expression, argument);

  try {
    await page.waitForFunction(() => window.__anny_editor && window.__anny_editor.ready === true, undefined, {
      timeout: 600000,
    });
  } catch (error) {
    const diagnostic = await page
      .evaluate(() =>
        window.__anny_editor
          ? { status: window.__anny_editor.status, error: window.__anny_editor.error, log: window.__anny_editor.log }
          : 'no automation surface: the module never evaluated',
      )
      .catch(() => 'evaluate failed');
    record('boot', {
      ok: false,
      diagnostic,
      failedRequests: results.failedRequests,
      httpErrors: results.httpErrors,
      pageErrors: results.pageErrors,
      console: results.console.slice(-10),
    });
    await page.screenshot({ path: path.join(ROOT, 'output/editor-screenshot.png') }).catch(() => {});
    process.exitCode = 1;
    return;
  }
  record('boot', { ok: true, status: await editor(() => window.__anny_editor.status) });
  // The neutral state, so the screenshot at the end shows the editor as it opens rather than wherever
  // the clip and the randomisation left it.
  const initialState = await editor(() => window.__anny_editor.saveState());

  // 1. The panels are built from the model's own description.
  const describe = await editor(() => {
    const d = window.__anny_editor.describe();
    return {
      bones: (d.bone_labels || []).length,
      phenotype: (d.phenotype_labels || []).length,
      local: (d.local_change_labels || []).length,
      face: (d.facial_action_labels || []).length,
      sample: {
        phenotype: (d.phenotype_labels || []).slice(0, 3),
        bones: (d.bone_labels || []).slice(0, 3),
      },
    };
  });
  record('describe', {
    // The panels are built from whatever the model reports, so a group this model does not expose is
    // not a failure: `ci-model.safetensors` carries phenotype blendshapes and no face or local ones.
    ok: describe.bones > 0 && describe.phenotype > 0,
    ...describe,
  });

  // 2. ...and the DOM has a control for each of them.
  const panels = await editor(() => ({
    inputs: document.querySelectorAll('#panels input').length,
    ranges: document.querySelectorAll('#panels input[type=range]').length,
    boneRows: document.querySelectorAll('[data-bone]').length,
    firstLabel: (document.querySelector('#panels input')?.closest('label, .control, .row, div')?.textContent || '')
      .trim()
      .slice(0, 40),
  }));
  record('panels', {
    ok: panels.inputs >= describe.phenotype + describe.local + describe.face,
    ...panels,
    expectedAtLeast: describe.phenotype + describe.local + describe.face,
  });

  // 3. A slider in the panel is wired end to end.
  //
  // The value is swept rather than fixed: `ci-model.safetensors` applies `gender` in steps, so 0.5 and
  // 0 produce identical vertices while 0.75 does not. A single value would test the model's response
  // curve, not the wiring, which is what this check is for — `phenotypeAfterEvent` proves the slider
  // wrote the parameter, and a change at any value proves the panel applies it.
  const beforeSlider = await editor(() => window.__anny_editor.digest());
  const valuesTried = [0.25, 0.5, 0.75, 1];
  let sliderChangedAt = null;
  for (const value of valuesTried) {
    const slider = await page.evaluate(amount => {
      const input = document.querySelector('#panels input[type=range]');
      if (!input) return null;
      const label = (input.closest('label, .control, .row, div')?.textContent || '').trim().slice(0, 40);
      input.value = String(amount);
      input.dispatchEvent(new Event('input', { bubbles: true }));
      return { label, value: input.value };
    }, value);
    if (!slider) {
      record('slider-moves-mesh', { ok: false, error: 'no range input in #panels' });
      break;
    }
    await page
      .waitForFunction(digest => window.__anny_editor.digest() !== digest, beforeSlider, { timeout: 4000 })
      .catch(() => {});
    if ((await editor(() => window.__anny_editor.digest())) !== beforeSlider) {
      sliderChangedAt = value;
      break;
    }
  }
  const sliderState = await editor(() => window.__anny_editor.parameters());
  record('slider-moves-mesh', {
    ok: sliderChangedAt !== null,
    changedAt: sliderChangedAt,
    valuesTried,
    before: beforeSlider,
    after: await editor(() => window.__anny_editor.digest()),
    phenotypeAfterEvent: sliderState.phenotype_kwargs,
  });

  // The same change through the editor's own parameter entry point. If this moves the mesh while the
  // slider above does not, the wiring is at fault rather than the model or the parameter itself.
  // The sweep above drives the first phenotype control; the API path drives the same parameter.
  const sliderLabel = describe.sample.phenotype[0];
  const beforeApi = await editor(() => window.__anny_editor.digest());
  await editor(name => window.__anny_editor.setParameter('phenotype', name, 0.75), sliderLabel);
  const afterApi = await editor(() => window.__anny_editor.digest());
  record('parameter-moves-mesh', {
    ok: afterApi !== beforeApi,
    label: sliderLabel,
    before: beforeApi,
    after: afterApi,
  });

  // 4. Posing goes through the pose session, not through a re-evaluate.
  const boneLabels = await editor(() => window.__anny_editor.describe().bone_labels || []);
  const bone = boneLabels.find(name => name && name !== 'root') || boneLabels[0];
  const beforePose = await editor(() => window.__anny_editor.digest());
  await editor(name => window.__anny_editor.setPose(name, { x: 35 }), bone);
  const afterPose = await editor(() => window.__anny_editor.digest());
  const poseStats = await editor(() => window.__anny_editor.stats());
  record('pose-through-session', {
    ok: afterPose !== beforePose && poseStats.source === 'session',
    bone,
    before: beforePose,
    after: afterPose,
    source: poseStats.source,
    lastMs: poseStats.lastMs,
  });

  // 5. Undoing the pose returns the exact same geometry.
  await editor(name => window.__anny_editor.setPose(name, { x: 0 }), bone);
  const backToRest = await editor(() => window.__anny_editor.digest());
  record('pose-reset-exact', { ok: backToRest === beforePose, before: beforePose, after: backToRest });

  // 6. GLB export, validated by the official Khronos validator.
  const exported = await editor(() => ({
    bytes: window.__anny_editor.exportGlb().bytes,
    magic: Array.from(window.__anny_editor.lastGlb.slice(0, 4)),
  }));
  const base64 = await editor(() => {
    const bytes = window.__anny_editor.lastGlb;
    let out = '';
    const step = 0x8000;
    for (let i = 0; i < bytes.length; i += step) out += String.fromCharCode.apply(null, bytes.subarray(i, i + step));
    return btoa(out);
  });
  fs.writeFileSync(EXPORT, Buffer.from(base64, 'base64'));
  const report = await validateBytes(new Uint8Array(fs.readFileSync(EXPORT)), {});
  record('export-glb', {
    ok:
      exported.magic.join() === '103,108,84,70' &&
      exported.bytes > 1e6 &&
      report.issues.numErrors === 0 &&
      report.info.hasSkins === true,
    magic: String.fromCharCode(...exported.magic),
    bytes: exported.bytes,
    validatorErrors: report.issues.numErrors,
    validatorWarnings: report.issues.numWarnings,
    hasSkins: report.info.hasSkins,
    vertices: report.info.totalVertexCount,
    triangles: report.info.totalTriangleCount,
    materials: report.info.materialCount,
  });

  // 7. Save/load round trip: the geometry and material come back exactly.
  const savedState = await editor(() => window.__anny_editor.saveState());
  const savedDigest = await editor(() => window.__anny_editor.digest());
  await editor(() => window.__anny_editor.randomise(7));
  const randomised = await editor(() => window.__anny_editor.digest());
  await editor(state => window.__anny_editor.loadState(state), savedState);
  const loaded = await editor(() => window.__anny_editor.digest());
  record('state-round-trip', {
    ok: loaded === savedDigest && randomised !== savedDigest,
    saved: savedDigest,
    randomised,
    loaded,
  });

  // 8. The same seed randomises to the same character.
  await editor(() => window.__anny_editor.randomise(11));
  const seedEleven = await editor(() => window.__anny_editor.digest());
  await editor(() => window.__anny_editor.randomise(11));
  const seedElevenAgain = await editor(() => window.__anny_editor.digest());
  await editor(() => window.__anny_editor.randomise(12));
  const seedTwelve = await editor(() => window.__anny_editor.digest());
  record('randomise-deterministic', {
    ok: seedEleven === seedElevenAgain && seedTwelve !== seedEleven,
    seed11: seedEleven,
    seed11Again: seedElevenAgain,
    seed12: seedTwelve,
  });

  // 9. A texture, fetched by the page and applied to the material.
  await editor(async () => {
    const response = await fetch('../../output/editor-texture.png');
    window.__anny_editor.setTexture(new Uint8Array(await response.arrayBuffer()), 'editor-texture.png');
  });
  await page
    .waitForFunction(() => window.__anny_editor.texture() === 'editor-texture.png', undefined, { timeout: 15000 })
    .catch(() => {});
  const textureName = await editor(() => window.__anny_editor.texture());
  record('texture-applied', { ok: textureName === 'editor-texture.png', name: textureName });

  // 10. Clip playback moves the character.
  const clip = await editor(async () => {
    const response = await fetch('../../examples/motion.json');
    return window.__anny_editor.loadClip(await response.text());
  });
  const clipRest = await editor(() => window.__anny_editor.seekClip(0));
  const clipMoved = await editor(() => window.__anny_editor.seekClip(1));
  record('clip-playback', {
    ok: clip.frames >= 3 && clip.duration > 0 && clipMoved !== clipRest,
    clip,
    rest: clipRest,
    moved: clipMoved,
  });

  // 11. The viewport actually draws the character.
  const render = await editor(() => window.__anny_editor.renderInfo());
  const stats = await editor(() => window.__anny_editor.stats());
  record('viewport-renders', { ok: render.triangles > 20000, ...render, ...stats });

  // 12. Back to the neutral character, framed, for a screenshot and the record.
  await editor(state => {
    window.__anny_editor.loadState(state);
    window.__anny_editor.rebase();
  }, initialState);
  const shot = path.join(ROOT, 'output/editor-screenshot.png');
  await page.screenshot({ path: shot });
  fs.mkdirSync(path.join(ROOT, 'examples/editor/docs'), { recursive: true });
  fs.copyFileSync(shot, path.join(ROOT, 'examples/editor/docs/editor.png'));
  record('no-page-errors', {
    ok:
      results.pageErrors.length === 0 &&
      results.failedRequests.length === 0 &&
      results.httpErrors.length === 0,
    pageErrors: results.pageErrors,
    failedRequests: results.failedRequests,
    httpErrors: results.httpErrors,
  });

  results.ok = Object.values(results.checks).every(check => check.ok);
  save();
  const failed = Object.entries(results.checks)
    .filter(([, check]) => !check.ok)
    .map(([name]) => name);
  console.log(
    `\n${results.ok ? 'EDITOR QUALIFICATION PASSED' : 'EDITOR QUALIFICATION FAILED'}: ` +
      `${Object.keys(results.checks).length} checks` +
      (failed.length ? `, failed: ${failed.join(', ')}` : ''),
  );
  if (!results.ok) process.exitCode = 1;
}

main()
  .catch(error => {
    results.error = String((error && error.stack) || error);
    save();
    console.error(results.error);
    process.exitCode = 1;
  })
  .finally(async () => {
    if (browser) await browser.close().catch(() => {});
    if (server) server.close();
  });
