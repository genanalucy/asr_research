# Whisper / 跨语种 PEFT 适配方向（2026-10-14）

## 当前判断

这轮查新支持把目标从“小样本可行性”转到模型适配和调参。主实验应使用已核验的壮语全训练池 **55,805 条、107.40695 h**；20 h 子集只用于快速筛配方，最终模型与主结果回到全量训练池。父代理记录的 dev 为 7.84438 h / 2 人、test 为 5.52121 h / 2 人。不要把结构盘点中的120.77 h候选总时长当训练时数。

核心文献边界已经很清楚：Whisper 上的 LoRA、AdaLoRA、稀疏自适应秩、encoder/decoder 同时加 LoRA、新语言适配和相似供体 LoRA 初始化都有直接先例。新加一个 language ID、普通 LoRA、试若干学习率/秩，或只把方法换到壮语，都不构成算法首创。短期最稳妥的成果是用完整壮语训练池做受控适配比较，给出可复现的高数据量结果；方法新意应建立在强基线之后。

## 关键原文证据

| 来源 | 实际核读范围与数据规模 | 结果、限制及创新近邻 |
|---|---|---|
| Qian et al., *Learn and Don't Forget: Adding a New Language to ASR Foundation Models*, Interspeech 2024, DOI [10.21437/Interspeech.2024-1045](https://doi.org/10.21437/Interspeech.2024-1045), [arXiv:2407.06800](https://arxiv.org/abs/2407.06800) | 已读摘要、方法§2.1–2.5、数据/设置§3.1–3.4、结果表3–4和结论。FLEURS 未覆盖语种每语约12 h；Whisper large-v3。 | 比较全量微调、LoRA、soft language code、soft prompt 和 EWC。全量微调对新语言效果最好但损害原语言；只训练适配参数能保留旧语言能力，但新语言表现较弱。LoRA rank 8 用于 attention。证明“加新语种标识/LoRA”不是新意；若当前任务不要求保护旧语言，全量微调应作为强基线。全文、元数据、BibTeX：[`literature/additions_transfer_20261013/peft/2407.06800/`](../../literature/additions_transfer_20261013/peft/2407.06800/)。 |
| Song et al., *LoRA-Whisper: Parameter-Efficient and Extensible Multilingual ASR*, Interspeech 2024, DOI [10.21437/Interspeech.2024-892](https://doi.org/10.21437/Interspeech.2024-892), [ISCA 官方页](https://www.isca-archive.org/interspeech_2024/song24_interspeech.html) | 已读官方元数据/摘要、方法§3.1–3.3、数据和训练§4.1–4.2、语言扩展结果§4.3–4.5、表1–5及结论。Whisper-small；FLEURS 新语种训练为7.4–12.2 h，MLS donor 为103–247 h。 | §3.3 已提出 language-specific LoRA、相似语种 LoRA warm start 与 LoRA MoE；§4.4 按语言相似度挑 warm-start donor，错 donor 可比从头训练更差。作者报告 warm start 相比 full FT 相对改善23%、MoE 相比普通 LoRA 5%，仅是其设定下的结果。**这篇是本项目做 donor 初始化最直接的先例。**全文、元数据、BibTeX：[`literature/additions_transfer_20261013/peft/2024-interspeech-892/`](../../literature/additions_transfer_20261013/peft/2024-interspeech-892/)。 |
| Liu et al., *Sparsely Shared LoRA on Whisper for Child Speech Recognition*, [arXiv:2309.11756](https://arxiv.org/abs/2309.11756)（arXiv 版本；本轮未核正式会议版本） | 已读摘要、LoRA/AdaLoRA/S2-LoRA 方法、数据和设置、结果表1–2及结论。CSRC-2021 有28.6 h儿童普通话朗读和29.5 h会话；研究以28.6 h朗读集及1K/10K/30K条子集适配。 | 直接比较 LoRA、AdaLoRA、GLoRA、S2-LoRA 等。S2-LoRA共享低秩矩阵并学习稀疏 rank 系数，报告0.02%可训练参数，在域外条件优于常规 LoRA；AdaLoRA 已在 Whisper 上验证，动态秩不能作为本项目的新方法。普通话适配后粤语表现下降，作者讨论了共用 zh language ID 的 token 冲突。全文、元数据、BibTeX：[`literature/additions_transfer_20261013/peft/2309.11756/`](../../literature/additions_transfer_20261013/peft/2309.11756/)。 |
| Mondal et al., *Probing LoRA-to-LoRA Cross-Lingual Transfer for Unseen Low-Resource Conditions in Whisper-Based ASR*, Interspeech 2026, DOI [10.21437/Interspeech.2026-1133](https://doi.org/10.21437/Interspeech.2026-1133), [ISCA 官方页](https://www.isca-archive.org/interspeech_2026/mondal26_interspeech.html) | **标题、作者、出版信息及论文 PDF 已按 ISCA 正式档案核实；该官方记录未列 arXiv ID。**已读摘要、方法§3.1–3.3、实验设置§4.1–4.3、表2–4、结果§5和结论。Whisper-small/rank 32；donor 为 Hindi 600 h、Bengali 600 h、Marathi 550 h；recipient 条件含40、60、80、100 h。 | donor LoRA 初始化 recipient LoRA，再用 recipient 数据继续适配；用亲缘筛选和 token 分布 JS divergence 选 donor。作者报告比 recipient-only LoRA 相对 WER 下降9–17%；最大值来自40 h Bhojpuri。该论文已经覆盖 40–100 h recipient 和 orthographic donor selection，故“LoRA-to-LoRA 迁移/相似 donor”不能再主张为新意。它是正式会议论文，但本轮未复现，也未审代码；数字按论文表4转述。全文、元数据、BibTeX：[`literature/additions_transfer_20261013/peft/2026-interspeech-1133/`](../../literature/additions_transfer_20261013/peft/2026-interspeech-1133/)。 |
| Wang et al., *Low-Resource Speech Recognition by Fine-Tuning Whisper with Optuna-LoRA*, Applied Sciences 15(24), 13090 (2025), DOI [10.3390/app152413090](https://doi.org/10.3390/app152413090) | 本轮只核 Crossref 元数据和摘要；未读全文，数据小时数和完整参数搜索空间未确认。摘要涉及维吾尔语、哈萨克语、柯尔克孜语。 | 摘要报告相对 LoRA baseline 的 WER 降幅分别为20.98%、6.46%、8.72%，并用 validation loss + relative WER 双指标早停。由此 Optuna+LoRA 调参、早停不能作为首创。论文具体设置与结果表应在正式引用前核正文。 |
| Zhang et al., *AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning*, ICLR 2023, [arXiv:2303.10512](https://arxiv.org/abs/2303.10512) | 通过上述 Whisper S2-LoRA 原文的相关工作和对比表核到 AdaLoRA 的自适应秩设定；本轮没有重新下载/独立核读 AdaLoRA 全文。 | 是动态分配各权重矩阵 rank budget 的通用 PEFT 方法；且 S2-LoRA 已把自适应 rank 用到 Whisper 语音识别。把 AdaLoRA 搬到壮语可作工程验证，不能称 rank allocation 本身为新方法。 |
| Liu et al., *DoRA: Weight-Decomposed Low-Rank Adaptation*, ICML 2024, [arXiv:2402.09353](https://arxiv.org/abs/2402.09353) | 本轮核 arXiv 官方元数据和摘要，未读全文。摘要覆盖 LLaMA、LLaVA、VL-BART 等，不是 ASR 实验。 | 可作为 LoRA 的候选强基线，但现有摘要证据不能证明 DoRA 对 ASR/Whisper 的收益；要在壮语上试属于验证一个新组合，不等于发明 DoRA。 |

此外，仓库已有 XLS-R、MMS、Whisper 基础论文和低资源迁移材料（见 [`literature/README.md`](../../literature/README.md)、[`literature/reading-notes/core/2402.02302_acoustic_transfer_selection.md`](../../literature/reading-notes/core/2402.02302_acoustic_transfer_selection.md)）。XLS-R/MMS 是 encoder/CTC 路线，新增语种通常需要适配 CTC 输出词表/头并评估是否继续自监督预训练；这与 Whisper decoder 词表、语言提示和输出嵌入问题不同。当前检索没有核实到 w2v-BERT 上针对壮语式新语种适配、层冻结或 DoRA 的直接原文结果，不能把 Whisper 上的证据外推成 w2v-BERT 的已知收益。w2v-BERT 若加入，应作为模型族对照，按其具体 checkpoint 的输出目标和 tokenizer 单独实现与评测。

收尾补充：Optuna-LoRA 的22页全文已下载、核签名/页数/SHA256，主代理核对首页摘要；上表仍保留正文未核读的边界，不将下载状态提升为全文阅读。FCCA 的5页全文也已保存，详见 `transfer-download-status.md`。

## 三个起跑方案

1. **先做 107.40695 h 壮语全量池上的 Whisper 强基线。**用已有流程固定 full fine-tuning，再用同一数据、解码和训练/验证定义跑 LoRA；按 dev 决策，test 只用于冻结后的最终比较。小样本 5 h 结果只能说明工程试跑可行，不能代表 full-pool 配方。一个 20 h proxy 可用于快速筛秩、注入模块、学习率和步数，但候选定下后必须回到完整107 h训练并报告。
2. **做一轮受控参数效率矩阵。**至少比较 uniform LoRA 与 AdaLoRA/S2-LoRA 或 DoRA；在固定 full-pool 主实验中比较 encoder-only、decoder-only、encoder+decoder。分层学习率、逐层解冻可作为第二阶段因素，保持数据、总更新步数/训练轮数、dev 调参预算一致。新意首先是受控证据和层级/秩分配诊断，不是“首次用了某适配器”。
3. **分析输出 token/正字法瓶颈后再决定 tokenizer 适配。**先报告壮文字符/音节序列在 Whisper tokenizer 中的分片长度、稀有组合覆盖和错误类型；再比较固定 tokenizer、可训练 language embedding/soft prompt，以及有原则初始化的新增 token embedding/输出行。新 language embedding 已有 Qian 等先例，新增 token 行也只能在它确实改善壮文低频正字法并跨说话人稳定时作为贡献证据。XLS-R/CTC 则应另行检查新 CTC 词表的尺寸和随机/迁移初始化。

## 24–72 h 初结果窗口

根据仓库 5 h、1500 steps、单卡约23 min的实测，一天内可以拿到小规模 proxy 的初筛结果；但全量107 h读数、长音频比例、验证耗时、Whisper 模型大小和 LoRA checkpoint 保存会改变时间，不能按小时或样本量线性外推。四张 L20 适合并行跑少量互不冲突的配置，不能把4倍GPU直接等同单模型训练提速4倍。

- **约24 h：**跑通 20 h proxy 上 full FT 与 LoRA 两个对照，筛出稳定的学习率/更新预算，并核对按说话人分层的 dev 变化；这是选配置，不是最终结论。
- **约48 h：**将选定配置在实际107.40695 h训练池上完成至少一轮 full-pool 训练；并行完成一项高价值消融，例如 encoder-only 对 encoder+decoder，或 uniform LoRA 对 AdaLoRA。若 full-pool epoch/长音频使步数估计不符，先用真实吞吐量修正时间表。
- **约72 h：**补齐首批 PEFT 对照和受控复跑，输出完整开发集 CER、字符级分析和逐人指标；句级置信区间仅描述已观察话语。test 需等搜索、消融与配方冻结后，按预定矩阵评估，不因72小时截止而提前使用。当前 dev/test 各两名说话人，不能据此给出可靠的人群泛化区间。预期是得到方向性初结果，不承诺显著优势。

上述时窗是根据现有23分钟小试和四卡条件做的工程估计，不含数据人工听校、超长/损坏音频复核、模型下载或远程排队；大模型结果以首个 full-pool throughput 实测重新估算。

## 创新边界与风险

- **系统性实证贡献候选：**经来源和语种确认的北部武鸣壮语，107 h 真实训练池，多说话人拆分，容量受限 ASR foundation model 的 full FT/LoRA/自适应秩/模块位置可复现比较，并公开匿名化数据统计、配方与误差类型。主张应写成数据规模、适配策略比较或语言资源成果；论文是否符合目标会议或期刊要求，需要结合结果与投稿定位判断。
- **有机会形成方法贡献的方向：**若层/秩选择能由独立 dev 数据估计的语言特征或层梯度信号确定，并且在≥107 h full-pool 的强基线、多个 seed/说话人评估中稳定优于已有 AdaLoRA/S2-LoRA/LoRA，需要再对照近邻文献并查新；目前不能先宣称新方法。
- **瑶语先留作外部验证：**约37.83 h结构候选存在大量冲突、来源重叠和身份别名未决，当前身份未认证。未清理前不能把它混入壮语训练或用来声称跨语迁移收益；来源核验后可单独作外部迁移验证。
- **主模型选择：**若目标是只交付壮语模型，2024 Qian 的结果意味着 full FT 必须保留为性能基线；若目标是同一底模保留多语能力，才将遗忘控制和 language-specific adapters 设为核心研究问题。
