import { mkdir } from "node:fs/promises";
import { join } from "node:path";

const [root, manifestPath, batchName = "initial"] = process.argv.slice(2);
if (!root || !manifestPath) throw new Error("Usage: bun download.ts <new-collection-directory> <manifest.json> [batch-name]");
const papers = await Bun.file(manifestPath).json();

function decode(text: string) {
  return text.replace(/&amp;/g, "&").replace(/&quot;/g, '"').replace(/&#39;|&apos;/g, "'").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/\\=\{?([aeiouAEIOU])\}?/g, (_, vowel) => `${vowel}̄`).normalize("NFC");
}
function metas(html: string, name: string) {
  return [...html.matchAll(/<meta\s+name="([^"]+)"\s+content="([^"]*)"\s*\/?\s*>/g)]
    .filter(m => m[1] === name).map(m => decode(m[2]!));
}
async function command(argv: string[], timeout = 90000) {
  const proc = Bun.spawn(argv, { stdout: "pipe", stderr: "pipe", cwd: import.meta.dir });
  const timer = setTimeout(() => proc.kill(), timeout);
  try {
    const [stdout, stderr, code] = await Promise.all([new Response(proc.stdout).arrayBuffer(), new Response(proc.stderr).text(), proc.exited]);
    return { bytes: new Uint8Array(stdout), stderr, code };
  } finally { clearTimeout(timer); }
}
async function download(p: any) {
  const dir = join(root, `${p.id ?? p.key}_${p.key}`);
  await mkdir(dir); // Refuse to reuse any existing directory or overwrite a previous download.
  const page = p.page ?? `https://arxiv.org/abs/${p.id}${p.version ?? ""}`;
  const pdfUrl = p.pdf ?? `https://arxiv.org/pdf/${p.id}${p.version ?? ""}`;
  const record: any = { ...p, page, pdfUrl, directory: dir, retrievedAt: new Date().toISOString(), downloaded: false };
  try {
    const landing = await command(["curl", "-f", "-L", "--max-time", "45", "--retry", "1", "--retry-all-errors", "-sS", page], 100000);
    if (landing.code !== 0 && !p.metadataVerified) throw new Error(`Cannot retrieve paper page: ${landing.stderr.trim()}`);
    if (landing.code !== 0) record.pageWarning = `页面下载失败；使用已独立核实的题名与作者元数据：${landing.stderr.trim()}`;
    const html = landing.code === 0 ? new TextDecoder().decode(landing.bytes) : "";
    const title = metas(html, "citation_title")[0] ?? p.expectedTitle;
    if (!title) throw new Error("No verified title available");
    if (p.expectedTitle && title.replace(/\s+/g, " ").trim().toLowerCase() !== p.expectedTitle.replace(/\s+/g, " ").trim().toLowerCase()) {
      throw new Error(`Title mismatch: page reports ${title}`);
    }
    record.title = title;
    const pageAuthors = metas(html, "citation_author");
    record.authors = pageAuthors.length ? pageAuthors : (p.authors ?? []);
    record.date = metas(html, "citation_date")[0] ?? p.date;
    record.doi = metas(html, "citation_doi")[0] ?? p.doi;
    const result = await command(["curl", "-f", "-L", "--max-time", "90", "--retry", "1", "--retry-all-errors", "-sS", pdfUrl], 200000);
    if (result.code !== 0) throw new Error(`Cannot download PDF: ${result.stderr.trim()}`);
    const signature = new TextDecoder().decode(result.bytes.slice(0, 1024));
    if (!signature.includes("%PDF-")) throw new Error(`Response is not a PDF (${result.bytes.length} bytes)`);
    const pdfPath = join(dir, "paper.pdf");
    await Bun.write(pdfPath, result.bytes);
    record.pdfPath = pdfPath;
    record.bytes = result.bytes.length;
    record.sha256 = new Bun.CryptoHasher("sha256").update(result.bytes).digest("hex");
    const info = await command(["pdfinfo", pdfPath], 15000);
    if (info.code !== 0) throw new Error(`PDF parser failed: ${info.stderr.trim()}`);
    const infoText = new TextDecoder().decode(info.bytes);
    record.pages = Number(infoText.match(/^Pages:\s+(\d+)/m)?.[1] ?? 0);
    const firstPage = await command(["pdftotext", "-f", "1", "-l", "1", pdfPath, "-"], 15000);
    const text = new TextDecoder().decode(firstPage.bytes);
    const compact = (s: string) => s.normalize("NFKC").toLowerCase().replace(/[^\p{L}\p{N}]/gu, "");
    record.firstPageTitleMatch = firstPage.code === 0 && compact(text).includes(compact(title));
    await Bun.write(join(dir, "first-page.txt"), text);
    record.downloaded = true;
    await Bun.write(join(dir, "README.md"), `# ${title}\n\n- 作者：${record.authors.length ? record.authors.join("; ") : "见论文首页（未自动提取）"}\n- 日期：${record.date ?? "见原文"}\n- 状态：${record.status}\n- 论文页：${page}\n- 下载链接：${pdfUrl}\n- 优先级：${record.priority}\n- 用途：${record.use}\n- PDF：${record.pages}页；${record.bytes}字节\n- 首页题名自动核对：${record.firstPageTitleMatch ? "通过" : "需要人工核对（不代表PDF错误）"}\n- SHA256：${record.sha256}\n\n这些用途说明是调研判断，不是原论文结论的替代。\n`);
  } catch (error) {
    record.error = String(error);
    await Bun.write(join(dir, "DOWNLOAD_FAILED.txt"), `${p.expectedTitle ?? p.key}\n${page}\n${pdfUrl}\n${record.error}\n`);
  }
  await Bun.write(join(dir, "metadata.json"), JSON.stringify(record, null, 2));
  console.log(JSON.stringify({ title: record.title ?? p.expectedTitle, downloaded: record.downloaded, pages: record.pages, firstPageTitleMatch: record.firstPageTitleMatch, error: record.error }));
  return record;
}
// Limit concurrency to avoid overloading public paper servers.
const results: any[] = [];
for (let i = 0; i < papers.length; i += 2) {
  results.push(...await Promise.all(papers.slice(i, i + 2).map(download)));
}
await Bun.write(join(root, `download-results-${batchName}.json`), JSON.stringify(results, null, 2));
console.log(`Downloaded ${results.filter(r => r.downloaded).length}/${results.length} PDFs into ${root}`);
