# ASR speaker diversity 检索记录（2026-10-13）

## 检索范围与判据

检索 arXiv、ACL Anthology、Crossref/出版商公开页面，时间范围 2020–2026。保留条件是论文必须研究 ASR 训练数据组成、speaker-disjoint/held-speaker 评测、或在固定标注预算下显式控制 speaker/content；speaker recognition、纯 diarization 和只讨论说话人条件化的论文剔除。摘要或全文未明确给出人数/小时/文本控制的，标为“仅摘要/局部证据”，不作完整实验断言。

## 首轮筛选（10篇）

| 优先级 | 论文与官方链接 | 证据覆盖 | 与当前问题的关系 | 小时/人数/文本/域控制与局限 |
|---|---|---|---|---|
| A | Yong et al. 2025, *Effects of Speaker Count, Duration, and Accent Diversity on Zero-Shot Accent Robustness in Low-Resource ASR*；[arXiv](https://arxiv.org/abs/2506.04364) · [Bib](https://arxiv.org/bibtex/2506.04364) | 全文5页；本批已下载 | 最直接：固定总时长改人数，固定人数改人均时长，并固定 N/t 改 accent K | 英/西/普通话；T=5–15h（普通话至11.25h），N=5/10/15或最多25；表2、图1–4。文本 prompt/unique token 未严格配对，accent/domain 与 speaker 仍有混杂，测试为未见口音而非单纯未见 speaker。 |
| A | Berrebbi et al. 2023, *More Speaking or More Speakers?*；[arXiv](https://arxiv.org/abs/2211.00854) · [DOI](https://doi.org/10.1109/ICASSP49357.2023.10096078) · [Bib](https://arxiv.org/bibtex/2211.00854) | 全文8页；本批已下载 | 系统比较 supervised、SSL、ST 中小时数与 speaker 数 | LibriSpeech 表3固定/改变标注和未标注小时、speaker；train/dev/test speaker 不重叠；unique words/types 也报告但文本与书籍/speaker耦合，非民族语言域。 |
| A | Liu, Spence & Prud'hommeaux 2023, *Investigating Data Partitioning Strategies for Crosslinguistic Low-Resource ASR Evaluation*；[ACL](https://aclanthology.org/2023.eacl-main.10/) · [Bib](https://aclanthology.org/2023.eacl-main.10.bib) | 全文9页；本批已下载（arXiv:2208.12888） | 直接支持 speaker-disjoint / session-disjoint 评测和不确定性 | 五个低资源语言；逐 speaker/session holdout、4:1 random、heuristic、adversarial；回归含 duration、pitch、intensity、token/type、OOV、perplexity。未操控训练人数。 |
| A- | Gody & Harwath 2023, *Unsupervised Fine-Tuning Data Selection for ASR Using Self-Supervised Speech Models*；[arXiv](https://arxiv.org/abs/2212.01661) · [DOI](https://doi.org/10.1109/ICASSP49357.2023.10095264) · [Bib](https://arxiv.org/bibtex/2212.01661) | 全文5页；本批已下载 | 固定10h转写预算下显式限制 speaker、gender、audiobook/topic，适合拆 speaker 与文本覆盖 | HuBERT/LibriSpeech；speaker/topic/text 共同变化，selection 研究不是纯人数因果；未见 speaker 结果未明示。 |
| A- | Maison & Estève 2023, *Some Voices are Too Common*；[arXiv](https://arxiv.org/abs/2306.03773) · [DOI](https://doi.org/10.21437/Interspeech.2023-2124) · [Bib](https://arxiv.org/bibtex/2306.03773) | 全文5页；本批已下载 | 固定约50,000 utterances/71±3h，尽量匹配598 speakers，另设 high-diversity 条件 | French Common Voice/wav2vec2；表2–3、图3。speaker、accent、录音质量和文本质量仍混杂，重点是公平性而非 OOD speaker。 |
| B+ | Shi et al. 2021, *Leveraging End-to-End ASR for Endangered Language Documentation: An Empirical Study on Yolóxochitl Mixtec*；[ACL](https://aclanthology.org/2021.eacl-main.96/) · [Bib](https://aclanthology.org/2021.eacl-main.96.bib) | 仓库已有全文12页 | 濒危语言基准：训练/验证/测试按 speaker 隔离；10/20/50h 曲线 | 3名测试/验证 speaker，约100h；表1、表5；对小时数有曲线但没有 speaker-count 正交实验，文本与自然对话域不可拆。 |
| B | Boulianne 2022, *Phoneme transcription of endangered languages...single speaker scenario*；[ACL](https://aclanthology.org/2022.findings-acl.180/) · [Bib](https://aclanthology.org/2022.findings-acl.180.bib) | 仓库已有全文8页 | 单 speaker 是低多样性边界案例，可作反例而非直接因果近邻 | 11种濒危语言，单 speaker、小时数/PER/模型比较；没有人数对照，不支持“更多 speaker”结论。 |
| B | Cheng et al. 2025, *Exploring the Impact of Data Quantity on ASR in Extremely Low-resource Languages*；[arXiv](https://arxiv.org/abs/2409.08872) · [Bib](https://arxiv.org/bibtex/2409.08872) | 仅摘要/元数据，未下载 | 数据量与跨语言选择近邻，speaker 不是主变量 | Amis/Seediq，SSL continued pretraining；摘要未给 speaker、文本或域的正交控制，暂不纳入直接证据。 |
| B | Dossou 2025, *Advancing African-Accented Speech Recognition: Epistemic Uncertainty-Driven Data Selection for Generalizable ASR Models*；[arXiv](https://arxiv.org/abs/2306.02105) · [Bib](https://arxiv.org/bibtex/2306.02105) | 仅摘要/元数据，未下载 | accent OOD 与主动数据选择近邻，可参考 domain/generalization 评测 | 摘要称多域、多数据集和较少标注；未明确 speaker/hour/text 配对，不能用来支持人数效应。 |
| B | Park, Ahmad & Hain 2022, *Unsupervised data selection for Speech Recognition with contrastive loss ratios*；[arXiv](https://arxiv.org/abs/2207.12028) · [Bib](https://arxiv.org/bibtex/2207.12028) | 仅摘要/元数据，未下载 | 固定10h选择和目标域匹配的 data-selection 邻近 | 摘要报告 TED/WSJCAM0 WER，但没有 speaker-diversity 或 speaker-disjoint 结论，不作主证据。 |

## 已取得全文与本地位置

- `literature/additions_2026_10_13_speaker/2506.04364/paper.pdf`（5页）
- `literature/additions_2026_10_13_speaker/2211.00854/paper.pdf`（8页）
- `literature/additions_2026_10_13_speaker/2212.01661/paper.pdf`（5页）
- `literature/additions_2026_10_13_speaker/2306.03773/paper.pdf`（5页）
- `literature/additions_2026_10_13_speaker/2208.12888/paper.pdf`（9页）

本批优先五篇均通过 arXiv PDF 签名检查；没有发生下载失败，因此未创建 `DOWNLOAD_FAILED.txt`。其余筛选论文仅做摘要/现有仓库全文核验，未重复请求 API；如需全文应换 ACL/出版商或作者公开版来源，避免 API 429。

## 对武鸣壮语设计的可执行控制

1. **正交小时/人数**：固定总时长（如5h）改变 speaker 数；固定 speaker 数（如8人）改变总时长和人均时长；每格使用同一训练文本 prompt 集或匹配 unique 字符/词数。
2. **文本覆盖记录**：每格报告 utterance 数、字符/词 token 数、unique token/type、prompt/topic 数和 transcript perplexity；必要时做相同 prompt 的配对子集。
3. **speaker/domain 隔离**：train/dev/test speaker-disjoint；同时做 session/domain-disjoint 或把 session/domain 分层，避免把麦克风、地点、说话风格差异误报为 speaker 效应。
4. **测试报告**：同时给随机 utterance split、speaker-disjoint、跨 session/domain 结果；跨所有可留出 speaker 做 speaker-bootstrap，而不是只选一个“代表性”测试人。
5. **语言迁移边界**：上述论文不能直接证明武鸣壮语的最佳人数或阈值；当前候选的 26人/120.77h 需要先确认每人时长、prompt/text、session/domain 和测试 speaker 身份，瑶语37.83h的 speaker 数仍待身份核验。

## 结构化阅读卡

详见 `literature/reading-notes/speaker-diversity/` 下五个以 arXiv ID 命名的 markdown 文件。每份都标明全文覆盖级别、页/表定位、官方 Bib、控制变量、局限和对当前实验的启发。
