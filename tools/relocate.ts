import { join } from "node:path";
const [root, oldRoot] = process.argv.slice(2);
if (!root || !oldRoot) throw new Error("Usage: bun relocate.ts <copied-literature-directory> <original-directory>");
function relocate(value: unknown): unknown {
  if (typeof value === "string") return value === oldRoot || value.startsWith(`${oldRoot}/`) ? root + value.slice(oldRoot.length) : value;
  if (Array.isArray(value)) return value.map(relocate);
  if (value && typeof value === "object") return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, relocate(item)]));
  return value;
}
let count = 0;
for await (const path of new Bun.Glob("**/*.json").scan({ cwd: root, absolute: true })) {
  const original = await Bun.file(path).json();
  const updated = relocate(original);
  if (JSON.stringify(original) !== JSON.stringify(updated)) {
    await Bun.write(path, JSON.stringify(updated, null, 2));
    count++;
  }
}
console.log(`Updated absolute paths in ${count} copied JSON files; originals were not modified.`);
const entries: any[] = [];
for await (const path of new Bun.Glob("*/metadata.json").scan({ cwd: root, absolute: true })) {
  entries.push(await Bun.file(path).json());
}
const pdfs = entries.filter(r => r.downloaded);
for (const entry of pdfs) {
  const bytes = new Uint8Array(await Bun.file(entry.pdfPath).arrayBuffer());
  const hash = new Bun.CryptoHasher("sha256").update(bytes).digest("hex");
  if (hash !== entry.sha256) throw new Error(`Copied PDF checksum mismatch: ${entry.pdfPath}`);
}
console.log(`Verified ${pdfs.length} copied PDF files against their original SHA256 values.`);
console.log(`Literature directory: ${root}`);
