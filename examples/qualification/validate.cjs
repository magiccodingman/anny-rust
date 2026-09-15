// Independent validator for actual files emitted by the native exporter.
const fs = require('node:fs/promises');
const path = require('node:path');
const validator = require('gltf-validator');
(async () => {
  const files = process.argv.slice(2);
  if (!files.length) throw new Error('Pass generated GLB paths.');
  for (const file of files) {
    const bytes = new Uint8Array(await fs.readFile(file));
    const report = await validator.validateBytes(bytes, {uri: path.basename(file), maxIssues: 1000});
    await fs.writeFile(file + '.validation.json', JSON.stringify(report, null, 2));
    console.log(JSON.stringify({file, errors: report.issues.numErrors, warnings: report.issues.numWarnings, infos: report.issues.numInfos}));
    if (report.issues.numErrors || report.issues.numWarnings) {
      console.error(JSON.stringify(report.issues.messages, null, 2));
      process.exitCode = 1;
    }
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
