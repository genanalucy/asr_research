# 核心全文精读综合：实验准备影响

## 阅读范围与证据边界

本次精读覆盖六篇本地 PDF 全文：2021 EACL Mixtec（12 页）、2407.13292 Iu Mien（5 页）、2026 AfricaNLP FER/TER（14 页）、2025 IJCNLP Faithful Transcription（10 页）、2607.04814 linguistic relatedness（15 页）和 2402.02302 ATDS（13 页）。均读到方法、数据、表/图、讨论和局限；数字均在对应精读卡中标注 PDF 页/表。没有把下载成功当作精读，也没有重新运行论文代码。

2026 AfricaNLP 和 2607.04814 的正式发表状态未在本次工作中另行核验；两篇卡片按全文报告并保留预印本/未来年份边界。当前文档只提炼实验设计影响，不预测本项目结果。

## 1. 数据 split 与测试域

1. **保留说话人/内容隔离。** Mixtec 的 YMC-EXP 由 34 名说话人、505 条录音组成，train/validation/test 为 92.46/4.01/2.52 h，明确无内容重叠（表 1，PDF p.4）；YMC-NT 只有 3 名说话人，并按 EGS、CTB、FEF 划 clean-dev、clean-test、noise-test。当前项目应至少保留 speaker-disjoint test，并把录音来源、设备、文体写入 manifest。
2. **不要把不同文字/域随机混合。** Faithful Transcription 对 9 种语言把 Bible、非 Bible Source A/B 分开训练并交叉测试；Bible 语料通常比非 Bible 大 5--10 倍，却在自然语音上更差（PDF pp.336--337）。Bible 与非 Bible 的词汇 overlap 很低，overlap 与 WER 的 Pearson 相关为 -0.711（图 5，PDF p.337）。因此壮语/瑶语应至少设置自然会话、朗读/经文或词表示例的域标签与 out-of-domain test。
3. **保留风格移位测试。** Uneme 将约 8 h 语料分为 4,729 train、880 validation、1,027 natural test、1,051 careful test；careful speech 只用于测试（表 2，PDF p.4）。mHuBERT-147+Transformer 的 natural WER/CER/FER/TER 为 .997/.461/.267/.402，careful 为 1.622/.837/.515/.510（表 4--5，PDF pp.7--8），说明逐词慢速并不必然更容易。当前方案应保留自然会话主测试，朗读作为显式域移位测试。

## 2. 文字重复、域迁移与转写目标

1. **同一语料的 underlying/surface 双目标值得保留。** Mixtec underlying 目标保留形态边界和补括号声调，surface 目标更贴近发音；Conformer WER 分别 16.0/16.1 与 16.6/16.3（表 4，PDF p.6）。这支持在壮语/瑶语同时评估规范正字法和表层/音系转写，而不是提前丢弃形态或声调标记。
2. **文字重复或狭窄文本会制造虚高。** Faithful Transcription 的 Bible TT ratio 低、词汇受限、说话人常为单一或少数个体；Bible-only 在 Bible test 上高分但跨域最差（PDF pp.333--337）。训练/测试不能让章节、模板句或同一说话人跨 split 泄漏；需报告词汇 overlap/类型数。
3. **混合训练不等于域适配。** Faithful Transcription 中 full Bible 或额外 5 h Bible 与非 Bible 直接拼接几乎系统性降低 WER，而 Bible continued pretraining 后再对非 Bible 微调才改善（图 3--4，PDF pp.336--337）。若当前项目有朗读或经文资源，应先把它当表示继续预训练候选，再用目标会话监督微调，并保留 target-only 对照。
4. **人工转写可以是系统组件。** Mixtec 的 NTC 在 clean-test 上 ROVER CER/WER 4.6/13.8，ROVER-Fusion2 总体 CER 5.7；noise-test 中新手单独 CER 8.4 反而优于融合（表 9，PDF p.9）。若本项目有初级转写，建议保留规则/投票融合基线和人工回退，不把 ASR 当成所有噪声域的替代者。

## 3. 模型配方与资源规模

1. **约 10 h 可做方法筛选，但必须报告说话人限制。** Iu Mien MightLJSpeech 为 9.7 h、9,761 句、单女性说话人；8:1:1 split、三次独立 10-fold 平均。Whistle-small phoneme FT WER 2.71，subword FT 2.95；wav2vec2-cv10 phoneme WER 2.76（表 2--3，PDF p.5）。这适合作为 10 h 级小规模配方筛选，不足以证明跨说话人泛化。
2. **音系目标与预训练目标要匹配。** Iu Mien 使用 54 phoneme（去非声调变音符后 44）和 500 BPE；phoneme head 复用已见 IPA embedding（PDF pp.3--5）。勉语预训练没有 tone supervision，论文将八声调建模留作未来工作；当前项目应单列 tone-aware ablation。
3. **从 10/20/50 h 画数据曲线。** Mixtec Conformer WER 从 10 h 的 39.1/39.2 降到 20 h 的 26.2/26.2、50 h 的 18.0/18.0、whole 16.0/16.1（表 5，PDF p.6）。建议最低预算包含 10、20、50 h 或按实际可用量设置等比例点，避免只报一个全量结果。
4. **预适配预算要和目标预算分开。** 2607.04814 在 1/10/70 h target 上比较 auxiliary sequential FT，full-model 总预算 1,020 GPU h，且 1 h target 后 related/unrelated/target-only 即基本等价（PDF pp.5--9）。这说明不能直接复制大型全矩阵；应优先 target-only、轻量 LoRA 和少量 donor 对照。
5. **CPT donor 选择有可执行但昂贵的先例。** 2402.02302 使用 10 h Punjabi + 60 h donor CPT、1 h supervised FT；Hindi WER 23.5，接近 70 h Punjabi 22.2，而不相似 Bengali 25.2（表 1，PDF p.6）。ATDS 与 WERR 的相关 r=0.89（表 2，PDF p.8），但每次 CPT 约 15 h、需 4×A100 40GB（附录 A，PDF pp.11--12）。当前项目只有在存在足够无标注 donor 且能承受小规模 CPT 时才应采用；语言亲缘不能代替 donor ranking。

## 4. 音系评测与错误分析

1. **WER/CER 必须与音系指标并报。** AfricaNLP 用 24 维 {-1,0,+1} feature map、NA-masked Needleman-Wunsch、FER 和 TER；Uneme 最佳 WER .997 但 FER .267、TER .402（表 4--5，PDF pp.7--8），说明词级失败仍可能保留大量音段信息。
2. **声调应单列错误。** Yoruba 最佳 TER .372、FER .151；Uneme downstep FER .565，均明显高于部分辅音/元音特征（表 3、5，PDF pp.7--8）。壮语/瑶语应至少按 tone category、声母/韵母和音节边界统计；没有可靠 G2P 时应报告指标不可用，而不是用字符距离冒充音系结论。
3. **实现成本需要提前计入。** AfricaNLP 的全量 feature maps 在附录表 8--11，G2P、复杂声母和语言特定声调规则是复现瓶颈；建议先做冻结 encoder + 小 decoder 诊断，再做 full fine-tune，分别回答表示是否已有音系信息和监督微调是否改善词级输出。

## 5. 建议的最小实验矩阵与预算顺序

1. **数据审计阶段：** speaker/content/domain-disjoint train/dev/test；列出小时、utterance、speaker、类型数、TT ratio、词汇 overlap；至少留自然会话 test 和朗读/经文 OOD test。
2. **基线阶段：** target-only 的 grapheme/subword CTC 或 encoder-decoder；10/20/50 h 数据曲线；WER + CER；如有专家音系映射，再加 FER/TER。
3. **转写阶段：** underlying/surface 或规范/表层双目标；若有初级稿，增加规则/ROVER 融合并在噪声域单独评估。
4. **迁移阶段：** 先比较 target-only、related donor、unrelated donor；目标数据至少 1 h 和当前最大量各一个点。只有出现稳定收益，再做 ATDS 或 2-stage CPT；不把 2607.04814 的 relatedness 结论外推到联合训练。
5. **资源记录：** 每个配置记录 GPU 型号、小时、训练步数、随机种子和 checkpoint 选择；将 2402.02302 的 CPT 作为高成本上限参考，而不是默认预算。所有数字均应回链到具体表/页，禁止写预测结果。

## 未核对与不应外推

- 没有重新运行六篇论文的代码、数据或模型；论文中的公开性/可复现性只按 PDF 和本地 metadata 描述。
- 2026 AfricaNLP 与 2607.04814 的正式出版状态、数据许可和后续版本未在本次精读中核验。
- 六篇论文的 WER/CER/FER 不可直接横向比较：任务、文字规范、切分、模型训练阶段和测试域不同；综合文档只用于实验设计约束，不作为本项目结果基线。

