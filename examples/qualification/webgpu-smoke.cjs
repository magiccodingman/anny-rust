// Qualification for the browser GPU path: runs the WebGPU blendshape kernel in a
// real browser and checks it against the same f32 CPU reference that the native
// parity test uses.
//
// The full Chromium is launched in new-headless mode rather than Playwright's
// default headless shell, because the shell ships no GPU stack; Linux still needs
// `--enable-unsafe-webgpu` to let WebGPU see an adapter.
//
//   node examples/qualification/webgpu-smoke.cjs [module.js] [model.safetensors]
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');
const { chromium } = require('playwright');

const ROOT = path.resolve(__dirname, '../..');
const MODULE = process.argv[2] || 'output/wasm-web/anny_wasm.js';
const MODEL = process.argv[3] || 'output/ci-model.safetensors';
const RESULTS = path.join(ROOT, 'output/webgpu-result.json');
const MIME = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.wasm': 'application/wasm',
  '.safetensors': 'application/octet-stream',
};

const results = { module: MODULE, model: MODEL, checks: {}, console: [], pageErrors: [], httpErrors: [], ok: false };
let server;
let browser;

function save() {
  fs.mkdirSync(path.dirname(RESULTS), { recursive: true });
  fs.writeFileSync(RESULTS, JSON.stringify(results, null, 2));
}

function record(name, check) {
  results.checks[name] = check;
  save();
  console.log(`${check.ok ? 'ok  ' : 'FAIL'} ${name}: ${JSON.stringify(check)}`);
}

function serve() {
  return new Promise((resolve) => {
    server = http.createServer((request, response) => {
      const url = new URL(request.url, 'http://127.0.0.1');
      const file = path.join(ROOT, decodeURIComponent(url.pathname));
      if (!file.startsWith(ROOT) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
        response.writeHead(404).end('not found');
        return;
      }
      response.writeHead(200, { 'content-type': MIME[path.extname(file)] || 'application/octet-stream' });
      response.end(fs.readFileSync(file));
    });
    server.listen(0, '127.0.0.1', () => resolve(server.address().port));
  });
}

// Runs inside the page: import the wasm, build real (sparse) coefficients, then
// compare the WebGPU result against the f32 CPU reference.
const PROBE = async ({ moduleUrl, modelUrl, batch }) => {
  const mod = await import(moduleUrl);
  await mod.default();
  const exports = ['cpu_blendshapes', 'gpu_blendshapes', 'default_coefficients', 'gpu_adapter_name'].filter(
    (name) => typeof mod[name] === 'function',
  );
  const bytes = new Uint8Array(await (await fetch(modelUrl)).arrayBuffer());
  const coefficients = mod.default_coefficients(bytes, batch);
  const c = coefficients.length / batch;
  let nonZero = 0;
  for (let i = 0; i < coefficients.length; i += 1) if (coefficients[i] !== 0) nonZero += 1;
  const compare = (want, got) => {
    let worst = 0;
    let finite = true;
    for (let i = 0; i < want.length; i += 1) {
      const d = Math.abs(want[i] - got[i]);
      if (!(d <= worst)) worst = d;
      if (!Number.isFinite(got[i])) finite = false;
    }
    return { worst, finite };
  };
  const t0 = performance.now();
  const cpu = mod.cpu_blendshapes(bytes, coefficients, batch);
  const cpuMs = performance.now() - t0;
  const t1 = performance.now();
  const gpu = await mod.gpu_blendshapes(bytes, coefficients, batch);
  const coldMs = performance.now() - t1;
  const t2 = performance.now();
  const again = await mod.gpu_blendshapes(bytes, coefficients, batch);
  const warmMs = performance.now() - t2;
  return {
    navigatorGpu: !!navigator.gpu,
    exports,
    batch,
    c,
    size: cpu.length / batch,
    nonZero,
    activeShare: nonZero / coefficients.length,
    cpuMs,
    coldMs,
    warmMs,
    lengths: { cpu: cpu.length, gpu: gpu.length, again: again.length },
    first: compare(cpu, gpu),
    second: compare(cpu, again),
    adapter: mod.gpu_adapter_name(),
    cpuSum: cpu.slice(0, 1024).reduce((a, v) => a + v, 0),
    gpuSum: gpu.slice(0, 1024).reduce((a, v) => a + v, 0),
  };
};

async function launch() {
  const args = [
    '--enable-unsafe-webgpu',
    '--enable-features=Vulkan',
    '--no-sandbox',
    '--disable-dev-shm-usage',
  ];
  // The bundled playwright builds on this machine are only ever partial, so
  // prefer the installed Chrome — the same `CHROMIUM_PATH` override the editor
  // suite takes — and fall back to whatever playwright can launch.
  const attempts = [
    ['chrome channel', { channel: 'chrome', args }],
    ['system chrome', { executablePath: process.env.CHROMIUM_PATH || '/usr/bin/google-chrome', args }],
    ['bundled chromium', { channel: 'chromium', args }],
    ['playwright default', { args }],
  ];
  let last;
  for (const [label, options] of attempts) {
    try {
      const launched = await chromium.launch(options);
      results.launch = label;
      console.log(`launched via ${label}`);
      return launched;
    } catch (error) {
      last = error;
      console.log(`${label} unavailable: ${String(error.message).split('\n')[0]}`);
    }
  }
  throw last;
}

async function main() {
  const port = await serve();
  const origin = `http://127.0.0.1:${port}`;
  browser = await launch();
  results.browser = browser.version();
  const page = await browser.newPage();
  page.on('console', (message) => {
    if (message.type() !== 'error') return;
    results.console.push({ text: message.text(), url: message.location().url || '' });
  });
  page.on('response', (response) => {
    if (response.status() >= 400) {
      results.httpErrors.push({ url: response.url(), status: response.status() });
    }
  });
  page.on('pageerror', (error) => {
    results.pageErrors.push(String(error));
    save();
  });
  await page.goto(`${origin}/examples/editor/index.html`);
  const probe = await page.evaluate(PROBE, {
    moduleUrl: `${origin}/${MODULE}`,
    modelUrl: `${origin}/${MODEL}`,
    batch: 4,
  });
  results.probe = probe;
  save();

  record('webgpu-available', { ok: probe.navigatorGpu, navigatorGpu: probe.navigatorGpu });
  record('gpu-exports-present', { ok: probe.exports.length === 4, exports: probe.exports });
  record('real-coefficients-sparse', {
    ok: probe.nonZero > 0 && probe.activeShare < 0.1,
    nonZero: probe.nonZero,
    total: probe.batch * probe.c,
    activeShare: probe.activeShare,
  });
  record('cpu-reference-runs', {
    ok: probe.lengths.cpu === probe.batch * probe.size && Number.isFinite(probe.cpuSum),
    length: probe.lengths.cpu,
    batch: probe.batch,
    size: probe.size,
    cpuMs: probe.cpuMs,
  });
  record('gpu-agrees-with-cpu', {
    ok: probe.lengths.gpu === probe.lengths.cpu && probe.first.finite && probe.first.worst <= 1e-5,
    worst: probe.first.worst,
    adapter: probe.adapter,
    coldMs: probe.coldMs,
    cpuMs: probe.cpuMs,
    tolerance: 1e-5,
  });
  record('gpu-reproduces-on-second-run', {
    ok: probe.lengths.again === probe.lengths.cpu && probe.second.worst <= 1e-5,
    worst: probe.second.worst,
    warmMs: probe.warmMs,
  });
  // A missing favicon is the browser asking for a file the page never declares;
  // only real failures count.
  const noise = (entry) => /favicon\.ico/.test(entry.url || entry.text || '');
  const consoleErrors = results.console.filter((entry) => !noise(entry));
  const httpErrors = results.httpErrors.filter((entry) => !noise(entry));
  const ignored = results.console.length - consoleErrors.length + (results.httpErrors.length - httpErrors.length);
  record('no-page-errors', {
    ok: results.pageErrors.length === 0 && consoleErrors.length === 0 && httpErrors.length === 0,
    pageErrors: results.pageErrors,
    console: consoleErrors,
    httpErrors,
    ignoredFaviconRequests: ignored,
  });

  results.ok = Object.values(results.checks).every((check) => check.ok);
  save();
  const failed = Object.entries(results.checks)
    .filter(([, check]) => !check.ok)
    .map(([name]) => name);
  console.log(
    `\n${results.ok ? 'WEBGPU QUALIFICATION PASSED' : 'WEBGPU QUALIFICATION FAILED'}: ` +
      `${Object.keys(results.checks).length} checks` +
      (failed.length ? `, failed: ${failed.join(', ')}` : ''),
  );
  if (!results.ok) process.exitCode = 1;
}

main()
  .catch((error) => {
    results.error = String((error && error.stack) || error);
    save();
    console.error(results.error);
    process.exitCode = 1;
  })
  .finally(async () => {
    if (browser) await browser.close().catch(() => {});
    if (server) server.close();
  });