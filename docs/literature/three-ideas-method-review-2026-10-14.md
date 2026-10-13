# 三个方法候选的证据复核与实验设计（2026-10-14）

## 范围与结论

这是复用当前讨论上下文的咨询复核，不是 fresh/blinded 审查。只查阅本地近邻论文的方法、实验及相关结果段落，没有跑训练、读取测试数据或做新一轮穷尽网页查新。下述判定只表示通过本次早期近邻门，不代表全球首创认证。

已知数据：壮语 train 107.407 h / 22 speakers、dev 7.844 h / 2 speakers、test 5.521 h / 2 speakers；瑶语约 37.83 h 候选，语言变体和身份尚未认证。不能依靠未验证的瑶语说话人数量或假定壮瑶亲缘关系建立论文结论。旧 hours/speakers 方案不作为本文创新主线。

| 候选 | 本次判定 | 可辨认 delta | 主要风险 |
|---|---|---|---|
| 1 跨说话人稳定的固定预算 LoRA 分配 | PROCEED WITH CAUTION | 用训练内部未见 speaker 的实际适配收益及跨 speaker-group 不稳定性分配容量，而非训练梯度/谱能量 | 容易变成按 speaker folds 做昂贵超参搜索；单模块收益不一定可加 |
| 2 音频反事实可靠性加权蒸馏 | PROCEED（通过本次早期门） | 在同一真实前缀上比较匹配音频与错配音频的教师概率，只在得到声音支持且可靠的 token 上加强 KD | 反事实分数可能只是对错误输入的脆弱性；gold prefix 本身含强语言线索 |
| 3 保留供体 encoder、重置 decoder 的选择性迁移 | PROCEED WITH CAUTION | 拆解跨语迁移的声学与文本更新，再通过目标训练梯度选择供体 encoder 更新 | encoder-only/reset-head 有广泛先例风险；只有两语和来源差异难证明机制 |

本次未找到包含这些完整方法和结果的具体论文，因此没有 ABANDON 判定。候选 3 的创新最薄；候选 2 最有独立方法故事，但须先验证分数是否真正有用；候选 1 最适合先做低成本可行性 pilot。

## 近邻全文核验

1. Qian et al., *Learn and Don't Forget: Adding a New Language to ASR Foundation Models*, arXiv:2407.06800v3，§2–4：比较 full FT、LoRA、soft language code、soft prompt、EWC，关心新语言表现与旧语言遗忘。支持 Whisper 未覆盖语言适配的基线设置，但本文未读到按 held-out speakers 估计 module/rank 泛化收益的方案。
2. Song et al., *LoRA-Whisper*, Interspeech 2024-892，§3.2–3.3：每语言在 encoder 和 decoder 加独立 LoRA；用 Whisper LID 确定近似语言，warm start 或 MoE。供体 LoRA 初始化、保留 base 与避免旧任务遗忘已经存在，不能作为我们的新点。
3. Mondal et al., *Probing LoRA-to-LoRA Cross-Lingual Transfer for Unseen Low-Resource Conditions in Whisper-Based ASR*, Interspeech 2026-1133，§3.1–3.3、§4：两阶段供体选择（语系限制后 tokenizer 分布 JS），供体 LoRA 初始化目标 LoRA。跨语顺序适配和使用低资源供体均不是新点；其 decoder 分布动机可作为候选 3 的对照，不意味着我们两种语言符合该前提。
4. *Sparsely Shared LoRA on Whisper for Child Speech Recognition*，arXiv:2309.11756，§2–4：分 Enc-SAM/Enc-FFM/Dec-SAM/Dec-CAM/Dec-FFM 五模块，共享低秩矩阵和稀疏 rank 系数；分析 AdaLoRA 的 rank allocation 与 regularization。本文已有模块差异、rank 分配和跨域泛化评估，因此“区分 encoder/decoder 并分配 rank”不能独立称创新。
5. Mishra et al., *Adaptive Fisher-Whitened Cross-Covariance for Low-Resource Speech Recognition*, arXiv:2609.29800，§2.1–2.3、§3–4：FCCA 固定 task-informed 两侧基底只训练小 core；AC-FCCA 跨层共享；AR-FCCA 在固定 core 参数预算下以 retained spectral energy 分配各矩阵 rank。train calibration 256 例、多 FLEURS 语言、Whisper medium / Qwen3-ASR；是候选 1 的强近邻，必须明确我们的选择依据是 held-out-speaker gain，不是又一种训练统计 rank heuristic。
6. Gandhi et al., *Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling*, arXiv:2311.00430，§3.3、§4.2、§5 及 architecture 描述：KL+序列伪标签、reference-WER 样本过滤；其学生复制并冻结完整 teacher encoder，缩减 decoder。medium→small 独立学生不等价于该结构。质量过滤、大小模型蒸馏、减少错误传播均有先例；候选 2 的差异是同 gold-prefix 的匹配/错配声音证据。
7. San et al., *Predicting positive transfer ... using acoustic pseudo-tokens*, arXiv:2402.02302，摘要、引言和方法起始：自监督 donor+target CPT、ATDS 声学伪 token 相似度预测供体收益。encoder 声学迁移和供体选择已存在，但这与监督 Whisper adapter 的 block 更新筛选不是同一个方法。
8. Florian et al., *Evaluating the Effect of Linguistic Relatedness ...*, arXiv:2607.04814v2，摘要、引言及实验框架：四大模型、两个语料、六因素、相关/不相关供体顺序适配；在其设置下有少量目标标签后亲缘并未给出实质收益。它是候选 3 的重要反证预期；不能预设跨语训练一定优于额外 target-only 更新。

这些论文目前证实“附近已有很多方法”，没有证实我们的具体 delta 已被实施。候选 2 仍需专项补检 acoustic-grounded/confidence/contrastive KD；候选 1 需补 OOD-aware/domain-generalization rank selection；候选 3 需补 encoder-only transfer、decoder reset、gradient-conflict update selection。

## 候选 1：以未见说话人收益分配固定 LoRA 预算

### 方法与可写主张

在 base Whisper-small 或 medium 上固定候选模块定义，选 8 个 disjoint 模块组。先画清每组具体矩阵、层范围、形状和单位 rank 参数数，避免“八组”任意选择。训练内部划成 3 个 leave-speaker-group-out folds；每 fold 只在其其余 speaker 训练局部 adapter，在留出 speaker 计算相对 base 的 token-normalized CE 改善，并补自由生成 CER。

每组分数 u_g = mean_fold(ΔL_g) − λ·sd_fold(ΔL_g)。选择 λ 只能用内部 folds，不用 test，留公开 dev 作方案确认而不是反复选模块。rank 的分配必须满足 Σ_g r_g·c_g ≤ B，其中 c_g 是该组 rank 增加 1 的真实参数成本，不是让各组 rank 相加就称“预算相同”。最终从同 base 初始化，在全部 107.407 h train 上重训。

候选主张：“在相同可训练参数预算下，针对未见 speaker 稳定收益的容量选择，比训练重要性或统一 rank 更能提升 speaker generalization。”这必须通过最终评测成立，不能仅凭 u_g 表格写出。

### 设计缺陷与修正

- 单组在小 rank 的收益不等于多组合并后的边际收益，模块可能互补/冲突。至少加逐组添加敏感性或 2–3 组交互检查；否则把分配称 heuristic，不称最优解。
- 各组可训练参数不同，探针需等参数预算/等更新预算；更大组自然有更多能力不是其泛化价值更高。
- 原始 CE、训练梯度范数容易受 token长度、层尺度和矩阵形状影响；按 token 聚合，gradient baseline 必须规范化。
- 三 folds 的 sd 估计很噪，须报告 fold 别结果、rank 稳定性、不同 speaker 分组重采样。不能将 sd称统计置信度。
- LoRA 初始化时 B=0，A梯度为0；基于初始参数梯度分配可能退化，至少在 warmup 后或 effective ΔW 空间估计。
- 使用 fold 的未见文本可能使分数主要反映文本覆盖。记录 train/heldout token/type、OOV、exact prompt overlap；共有seen样本太少时只做 novel-prompt/coverage-matched 检查，不强行固定交互矩阵。
- 24次局部训练（8组×3folds）是实际选择成本，不能仅报最终训练省参数。

### 实验

pilot：5–10 h 内部采样，8组×3folds同短预算；先检查分数有无可复现的spread、联合rank是否实现。主实验：5/20/full h，uniform LoRA、gradient allocation、AdaLoRA、候选；equal-number random module/rank search（同24探针次数和相同选择数据）作为“只是搜索得多”的强对照；FCCA/AR-FCCA可在实现可靠后补强，不伪称已复现。关键配置3seeds；以最终参数量和总GPU-hours两条公平轴报告。消融 λ=0、random folds、utterance-random代替speaker folds、单组分数与交互修正。

主指标 CER/WER、各speaker分数、worst-speaker、RTF/显存/总搜索成本；test仅最终方案冻结后使用。可接受否定：不优于同预算随机搜索；优势只在低小时出现；fold排名不稳定。这能揭示 heldout-loss 分配成本/边界，但没有稳定结果时方法贡献不成立。

## 候选 2：音频反事实支持的可靠性蒸馏

### 方法与可写主张

medium teacher先以真实 train 标签适配，small student从自身base开始；两者词表/tokenizer一致性必须逐项核验，不跨模型复制层。教师在同 gold prefix y_<t 上对真实 x 和两个 duration-matched、转写不相同的 train音频 x'_j 前向：

s_t = log p_T(y_t | x,y_<t) − mean_j log p_T(y_t | x'_j,y_<t)。

仅以训练音频生成反事实；排除同转写、哈希重复及疑似同录音片段；优先matched speaker/channel作为更难反事实，同时加异speaker随机作为敏感性。冻结teacher，detach权重。用校准过的 f(s_t) 与 token correctness/confidence 构成有界 w_t，真实标签 CE 全保留：L = CE_gold + α·T²·[Σ_t w_t KL(p_T^T || p_S^T) / Σ_t w_t]。需在有效KD权重全零时安全退回CE。special/pad token不参与gate。

候选主张：“教师对真实声音的相对支持，比仅置信度或转写准确率更能识别有益蒸馏token，从而降低错误继承，保持学生速度优势。”

### 核心缺陷与校验

- s_t不是因果音频依赖的完美估计。错误音频可能触发OOD/incongruence，任何正确token都大幅降概率。对照同speaker、相似长度/响度/词汇的难negative，并独立检查正确/错误teacher token的区分度。
- gold prefix给了完整真实历史，预测易由语言模型补全；低s_t但正确token并非无用，尤其词尾、功能词。对 gate与token位置/频率/长度的相关性分层，避免“丢掉易token”才改善。
- correctness应明确定义：在gold prefix下argmax是否gold，或自由生成对齐后的correctness；不能混用二者。本方法token KD采用gold prefix，WER filter另用free generation作为baseline。
- teacher confidence、correctness 与 s_t高度相关时，新分数不贡献独立信息；关键消融是仅correctness/confidence vs 加s_t，而不是只比无过滤KD。
- wrong音频会差异地改变prefix相容性，silence/null是更弱不同分布negative；wrong vs null消融不能单独证明音频 grounding。
- 全量三次teacher前向+词表logits存储昂贵；训练时在线KD或分片top-k概率缓存，top-k KL近似与完整KL差异需小样本核验。gate计算成本计入训练总账。

### 实验与可能结论

先20h pilot：无KD、standardKD、reference-WER过滤、token confidence过滤、correctness-only、等比例随机token过滤、候选；每个过滤法匹配保留token比例或有效KD权重，统一student预算。先在train内部speaker folds校准阈值，冻结后dev比自由生成CER。所有方法真实CE相同，不能把更多硬标签训练混入新方法优势。

主实验20/full h、3seeds；teacher medium作为性能上限，small学生报CER/WER、RTF、峰显存、参数量。错误继承以teacher自由生成错字附近的student错误变化及在gold prefix下teacher wrong-token上的student错误共同报告，别只挑成功样例。消融wrong两样本/一样本/null、只s_t/只confidence/联合、soft gate/hard filter、同speaker/异speakernegative。

支持结果：联合gate在同保留率下优于confidence/WER/random，且teacher错误区域的学生改善而非全靠易token。否定结果：s_t分辨力≈confidence、随机过滤同样好、只改善gold-prefix loss不改善free decoding；应将新gate判为无独立价值，仍保留完整KD benchmark结果。方法pilot通过前不宣称幻觉降低。

## 候选 3：隔离供体声学更新，避免跨语decoder迁移

### 方法与delta

供体监督适配后只继承encoder LoRA/effective update；目标decoder始终从base/零adapter开始，再用真实目标标签训练。诊断对照完整donor→target、encoder-only→target、decoder-only→target、target-only。进一步在目标train内部用微小校准集比较供体encoder各block的effective update ΔW_g与目标loss gradient G_g：一阶有利分数约为 −〈G_g,ΔW_g〉，保留稳定非负有益block；必须先明确符号及归一化。

梯度一致性若指donor gradient vs target gradient，需要说明它与已训练donor update不是同一个对象；训练路径改变后末步donor梯度未必描述累积更新。LoRA A/B因子有尺度/gauge自由度，应比较effective矩阵更新，不直接拼因子梯度cosine。目标encoder基底状态不同也会改变gradient，必须固定同一评估点。

这是 encoder/decoder迁移机制与block筛选候选，不能把“壮语帮瑶语”“相近语言有利”当创新。

### 实验与风险

语言/身份确认前只运行壮语为target、现有瑶语为来源受限donor的pilot，不宣称跨未见speaker泛化。必须检查瑶语来源、重复章节、文本规范和录音风格；donor几乎单人/圣经域的收益或损害可能只是域/身份效应。

目标1/5/20/full h，供体1/10/all h，先固定一个中等格（target5h、donor10h）做四种迁移诊断，再加入selected encoder与equal-block-count random selection。target-only同目标更新是第一对照；增加target-only额外epochs以及同总GPU预算对照。供体fullmodel/encoder-only的更新参数和训练预算不同要分别报告，而不能用同epochs假装同compute。所有方法target stage相同真实标签和更新预算；校准与目标fold划分同候选1，最终test单次。

可辨认delta需有结果支持：decoder迁移损害与token分布差异一致，同时encoder迁移保留收益，gradient筛选在等选择预算下优于简单encoder-only/random。只在两语一个方向成功足以成为case study，但不支持普适供体策略。若身份可核验，补反向迁移；否则用可验证公开供体形成第二donor，不把未验证身份写成多人数据。

可能否定：额外target epochs完全解释迁移收益；encoder-only无增益；gradient筛选不比随机；低标签有益但full107h无益。这与2607.04814结果并不矛盾，可报告“在该数据条件下，监督供体迁移没有省标注优势”，但缺乏正向方法差异时不应包装新迁移算法。

## 推进优先级

1. 优先做候选 2 的 gate诊断（已有teacher训练后，不训练整个学生即可量化score/confidence/correctness关系）与候选 1 的少量局部探针；两者是互不依赖的早期门。
2. 候选 1 先检验fold稳定性/搜索公平性，再扩展全量；候选 2 确认gate独立信息后跑20h matched controls。
3. 候选 3 最后且有身份前置，不让瑶语不确定性阻塞壮语主线。
4. 不为凑三创新同时全量搜索。pilot分别记实测GPU-hours和停止判据，再选择一条主方法及一条辅助验证。负结果必须保留，选最终test前冻结方案和主张。

四张L20可并行pilot，但上述候选尚未基于本地真实模型测速给出训练耗时；避免承诺精确天数或收益。
