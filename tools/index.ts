import { join, relative } from "node:path";
const [root] = process.argv.slice(2);
if (!root) throw new Error("Usage: bun index.ts <collection-directory>");
const all: any[] = [];
for await (const path of new Bun.Glob("*/metadata.json").scan({ cwd: root, absolute: true })) {
  all.push(await Bun.file(path).json());
}
const norm = (s: string) => s.normalize("NFKC").toLowerCase().replace(/[^\p{L}\p{N}]/gu, "");
const titleIds = new Map(all.filter(r => r.id).map(r => [norm(r.title ?? r.expectedTitle), r.id]));
const best = new Map<string, any>();
const rank = (r: any) => (r.downloaded ? 10 : 0) + (r.venue ? 5 : 0) + (r.firstPageTitleMatch ? 1 : 0);
for (const r of all) {
  const id = r.id ?? titleIds.get(norm(r.title ?? r.expectedTitle)) ?? r.key;
  if (!best.has(id) || rank(r) > rank(best.get(id))) best.set(id, r);
}
const order = ["2407.13292", "2406.02166", "2601.09050", "2610.13807", "cantonese_phoneme_augmentation", "2609.18194", "explicit_tone_bribri", "2407.06800", "2305.13516", "2212.04356", "2111.09296", "2006.11477", "tone_yoruba", "2402.02302", "2607.04814", "2606.31642", "2604.18204", "2404.03073", "2503.23542", "zhuang_prosody_svm_2011", "zhuang_cn_s2tt_2026"];
const records = [...best.entries()].sort((a, b) => order.indexOf(a[0]) - order.indexOf(b[0])).map(([, r]) => r);
const downloaded = records.filter(r => r.downloaded);
const missing = records.filter(r => !r.downloaded);
const pdfs = all.filter(r => r.downloaded);
for (const r of pdfs) {
  const bytes = new Uint8Array(await Bun.file(r.pdfPath).arrayBuffer());
  if (!new TextDecoder().decode(bytes.slice(0, 1024)).includes("%PDF-")) throw new Error(`Invalid PDF signature: ${r.pdfPath}`);
  const hash = new Bun.CryptoHasher("sha256").update(bytes).digest("hex");
  if (hash !== r.sha256 || bytes.length !== r.bytes || !r.firstPageTitleMatch) throw new Error(`Integrity or title check failed: ${r.pdfPath}`);
}
function bibEscape(s: string) {
  return s.replace(/\\/g, "\\textbackslash{}").replace(/([&%$#_])/g, "\\$1");
}
function bib(r: any) {
  if (r.verifiedBibtex) return r.verifiedBibtex;
  const title = r.title ?? r.expectedTitle;
  const year = String(r.date ?? "").match(/\d{4}/)?.[0] ?? (r.id ? `20${r.id.slice(0, 2)}` : "");
  const key = r.id ? `asr_${r.id.replace(/\W/g, "")}` : r.key.replace(/\W/g, "_");
  const fields = [`  title = {{${bibEscape(title)}}}`];
  if (r.authors?.length) fields.push(`  author = {${r.authors.map(bibEscape).join(" and ")}}`);
  if (year) fields.push(`  year = {${year}}`);
  if (r.venue) {
    fields.push(`  booktitle = {${bibEscape(r.venue)}}`);
    if (r.publisher) fields.push(`  publisher = {${bibEscape(r.publisher)}}`);
    if (r.volume) fields.push(`  volume = {${bibEscape(r.volume)}}`);
    if (r.publicationPages) fields.push(`  pages = {${bibEscape(r.publicationPages)}}`);
  } else if (r.id) {
    fields.push(`  eprint = {${r.id}}`, "  archivePrefix = {arXiv}");
  }
  if (r.doi) fields.push(`  doi = {${r.doi}}`);
  fields.push(`  url = {${r.page}}`, `  note = {${bibEscape(r.status + (!r.downloaded ? "；尚未取得全文，仅元数据/摘要核实" : ""))}}`);
  return `@${r.venue ? "inproceedings" : "misc"}{${key},\n${fields.join(",\n")}\n}\n`;
}
const bibHeader = "% 文献条目按实际下载/核实版本生成。除已核对的Whisper ICML正式版外，正式发表书目信息请投稿前再次核对。\n% arXiv条目不是虚构的会议正式版本；Unicode作者/题名建议使用支持UTF-8的BibLaTeX/Biber。\n\n";
await Bun.write(join(root, "references.bib"), bibHeader + downloaded.map(bib).join("\n"));
await Bun.write(join(root, "pending-fulltext.bib"), "% 以下论文尚未取得全文，不能声称已完成全文阅读。\n\n" + missing.map(bib).join("\n"));
await Bun.write(join(root, "SHA256SUMS.txt"), pdfs.map(r => `${r.sha256}  ${relative(root, r.pdfPath)}`).sort().join("\n") + "\n");
const missingLines = missing.map((r, i) => `## ${i + 1}. ${r.title ?? r.expectedTitle}\n\n- 作者：${(r.authors ?? []).join("; ")}\n- 年份：${String(r.date).slice(0, 4)}\n- DOI：${r.doi ?? "未提供"}\n- 论文页：${r.page}\n- 官方/登记PDF入口：${r.pdfUrl}\n- 失败原因：${r.error}\n- 用途：${r.use}\n- 核实范围：题名/作者/出版元数据；摘要按此前调研记录。未取得全文，不推断全文方法和结果。\n`).join("\n");
await Bun.write(join(root, "需要补充全文.md"), `# 需要补充的全文\n\n共${missing.length}篇。最优先补充粤语Phoneme-Aware Augmentation：这是拟研究声调/韵律保护增强的直接近邻。\n\n可通过学校图书馆、合法公开作者版或向作者索取；不需要购买或提供账户密码给本工具。\n\n${missingLines}`);
const table = records.map((r, i) => {
  const title = (r.title ?? r.expectedTitle).replace(/\|/g, "\\|");
  const status = r.downloaded ? `[PDF](${relative(root, r.pdfPath)}) · ${r.pages}页` : "待补全文";
  return `| ${i + 1} | ${title} | ${r.priority} | ${status} | ${r.use.replace(/\|/g, "\\|")} |`;
}).join("\n");
const readme = `# 壮语与瑶语ASR论文资料包\n\n整理日期：2026-10-12。\n\n- 已下载 **${downloaded.length}篇不同论文、${pdfs.length}份PDF**。Whisper公开预印本与ICML正式版属于同一篇，不重复计数；总目录优先指向ICML正式版。\n- **${missing.length}篇尚未取得全文**：见[需要补充全文](需要补充全文.md)。\n- 所有${pdfs.length}份PDF均检查了文件签名、可解析页数、首页题名和SHA256；这不等于已经逐篇精读全文。\n- 每篇PDF单独目录，附README、元数据和首页文本。历史失败目录保留供追溯；有成功重试的论文不列为待补。\n- [选题调研与实验建议](选题调研与实验建议.md)：候选方向、创新边界、实验设计和所需数据条件。\n- [references.bib](references.bib)：已取得全文的论文，可导入文献管理软件；正式出版信息请投稿前核对。\n- [pending-fulltext.bib](pending-fulltext.bib)：尚未取得全文的三篇，单独保存，不冒充已读文献。\n- [SHA256SUMS.txt](SHA256SUMS.txt)：文件校验值。\n\n## 建议阅读顺序与目录\n\nA表示选题/基线/近邻查新优先；B表示补充背景或特定路线的参考，不代表论文质量高低。以下用途是调研判断，不是原论文结论的替代。\n\n| 阅读顺序 | 题名 | 优先级 | 全文 | 用途 |\n|---|---|---|---|---|\n${table}\n\n## 引用与使用注意\n\n1. 下载的arXiv版本与正式发表版本不混同。已录用不等于已经核实最终页码、卷期。BibTeX按实际获取版本生成，不编造会议出版细节；Whisper ICML条目已核对PMLR官方页面。\n2. 声调辅助、音节分解、语言token、LoRA、LM重打分已有先例；换到壮语/瑶语不自动构成方法创新。声调保护增强也须精读粤语近邻全文后再定差异。\n3. 2026年Error Rate is Not Enough的声调结果来自Kakabe，不能直接外推到壮语/瑶语；相关亲缘迁移研究的结论是针对辅助语顺序预适配，不是否定所有多语言训练。\n4. 壮语—中文S2TT是翻译任务，不能把中文译文混作壮语原文ASR转写。Iu Mien不代表所有被称为“瑶语”的变体。\n5. 参考文献中部分新作仅为预印本；收录不代表其结果已经独立复现。\n\n## 资源与官方入口（不是额外论文）\n\n- MightLJSpeech勉语数据：https://github.com/mightmay/MightLJSpeech （单说话人朗读；用户是否同语种仍需确认）\n- Whistle/CAT实现入口：https://github.com/thu-spmi/CAT/tree/master/egs/cv-lang10\n- MMS官方说明与语种清单：https://github.com/facebookresearch/fairseq/tree/main/examples/mms\n- Whisper官方实现：https://github.com/openai/whisper\n- XLS-R/wav2vec实现：https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec\n\n本次未下载数据集、运行模型训练、上传用户数据或改动当前插件项目。\n`;
await Bun.write(join(root, "README.md"), readme);
await Bun.write(join(root, "literature-catalog.json"), JSON.stringify({ downloadedPapers: downloaded.length, pdfFiles: pdfs.length, missingPapers: missing.length, records }, null, 2));
console.log(JSON.stringify({ downloadedPapers: downloaded.length, pdfFiles: pdfs.length, missingPapers: missing.length, allPdfsVerified: true, missingTitles: missing.map(r => r.title ?? r.expectedTitle) }, null, 2));
