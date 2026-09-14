// Optional development qualification. Serves only repository files on loopback.
const fs = require('node:fs/promises');
const http = require('node:http');
const path = require('node:path');
const {chromium} = require('playwright');
const root = path.resolve(__dirname, '../..');
const mime = {'.html': 'text/html', '.js': 'text/javascript', '.wasm': 'application/wasm'};
let browser, server;
(async () => {
  server = http.createServer(async (request, response) => {
    try {
      const pathname = decodeURIComponent(new URL(request.url, 'http://127.0.0.1').pathname);
      const file = await fs.realpath(path.join(root, pathname));
      if (!file.startsWith(root + path.sep)) throw new Error('Path escapes test root');
      response.setHeader('Content-Type', mime[path.extname(file)] || 'application/octet-stream');
      response.end(await fs.readFile(file));
    } catch (_) { response.writeHead(404).end(); }
  });
  await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
  browser = await chromium.launch({headless: true, ...(process.env.CHROMIUM_PATH ? {executablePath: process.env.CHROMIUM_PATH} : {})});
  const page = await browser.newPage();
  const pageErrors = [];
  page.on('pageerror', error => pageErrors.push(String(error)));
  const model = process.argv[2] || 'output/ci-model.safetensors';
  const address = `http://127.0.0.1:${server.address().port}/examples/qualification/browser-smoke.html?model=${encodeURIComponent(model)}`;
  await page.goto(address);
  await page.waitForFunction(() => window.__anny_result != null, undefined, {timeout: 240000});
  const result = await page.evaluate(() => window.__anny_result);
  result.browser = browser.version();
  result.pageErrors = pageErrors;
  console.log(JSON.stringify(result, null, 2));
  await fs.mkdir(path.join(root, 'output'), {recursive: true});
  await fs.writeFile(path.join(root, 'output/browser-result.json'), JSON.stringify(result, null, 2));
  if (result.status !== 'passed' || pageErrors.length) process.exitCode = 1;
})().catch(error => { console.error(error); process.exitCode = 1; }).finally(async () => {
  if (browser) await browser.close();
  if (server) await new Promise(resolve => server.close(resolve));
});
