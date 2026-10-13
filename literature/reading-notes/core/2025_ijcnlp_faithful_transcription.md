# 论文精读卡

## 输入覆盖范围
- 来源：[paper.pdf](../../2025_ijcnlp_faithful_transcription/paper.pdf)，IJCNLP-AACL 2025 全文。
- 覆盖级别：全文（PDF 10 页；正文 pp.333--342；已读 §1--§6、图 1--5、表 1--10、附录实验说明）。
- 边界说明：Bible.is 数据抽取脚本和不可公开的 Bible ASR 数据未重新执行；数值按论文附录表读取。

## 标题
Faithful Transcription: Leveraging Bible Recordings to Improve ASR for Endangered Languages

## 链接
https://aclanthology.org/2025.ijcnlp-short.28/

## 研究问题
- Bible 朗读录音量大、对齐容易，但词汇窄、风格和说话人单一；它是否能泛化到当代自然语音（PDF pp.333--334）？
- 研究覆盖 5 个濒危语言和 4 个低资源语言，对比单域、跨域、混合训练与两阶段训练。

## 核心方法
- 统一用 XLSR-53 + CTC 头、训练 30 epochs、验证集选最佳、训练文本 trigram LM 解码（PDF p.335）。
- 对每种语言构造 Bible、非 Bible Source A/B；分别做单源训练、Bible+非 Bible 直接混合、5 h Bible 混合和 Bible 继续预训练后非 Bible 微调（图 1--4，PDF pp.334--337）。

## 模型 / 算法要点
- Amis/Tsou/Rukai/Swahili/Bambara/Iban/Fongbe 按 70/10/20 split；Quechua/Bribri 按 80/20 train/test；Bible dev/test 与非 Bible 规模匹配（PDF p.335）。
- 训练语料量差异大：Bible 新约通常 30+ h，是非 Bible 的 5--10 倍，但 speaker 数和文本 TT ratio 低（§5，图 5）。

## 实验设置
- 指标：跨 Source A/B/Bible 的 WER；图 1 比较 in-domain 到 out-of-domain 的相对退化；图 5 分析词汇重叠。
- 主要 baseline：Bible-only、另一非 Bible 源、同源 in-domain；混合训练与 2-phase。

## 主要结果
- Bible-only 在 Bible test 上高准确，但在当代非 Bible 语音上几乎系统性最差；Bible 数据即使大 5--10 倍也不保证泛化（§5，PDF pp.336--337）。
- 训练/测试词汇 overlap 与 WER 相关，图 5 给 Pearson r=-0.711、p<0.001；Bible 与非 Bible 测试 overlap 普遍最低（PDF p.337）。
- 直接把全部或 5 h Bible 与非 Bible 拼接通常使 WER 变差；Bible continued pretraining → 非 Bible fine-tune 的两阶段方案是唯一稳定改善策略（图 3--4，PDF pp.336--337）。
- 具体表例：Amis Source A 测试，Source A+Bible WER .348，2phase Bible+A .198；Bribri Source A，Source A+Bible .908，2phase .858；Quechua Source A，Source A+Bible .851，2phase .566（表 2、7、8，PDF p.341）。

## 优势
- 通过同语言多来源测试把“数据多”与“域匹配”拆开，结论横跨 9 种语言而非单一案例。
- 明确报告 train/test domain、词汇重叠和风格，能防止 Bible-only 高分被误读为真实泛化。

## 局限
- 论文承认无法确定受限词汇、speaker、戏剧化风格、内容哪一因素主导；Bible 说话人数常为一人但确切人数未给出（§5--§6，PDF pp.336--337）。
- XLSR-53 单一架构、短文篇幅；未系统测试声调/音系指标，也未报告运行方差。
- Bible 数据存在殖民和文化伦理背景，合法公开不等于适合所有语言复用（Ethical Considerations，PDF p.337）。

## 复现难点
- Bible.is 抽取和对齐脚本已因网站结构失效，且 Bible ASR 数据不能公开；只能复现非 Bible 端或重新抓取。
- 2-phase 需区分 continued representation training 与监督微调，学习率/步数未完全展开。

## 对当前研究的启发
- 训练/验证/测试应按来源与风格保留至少一个 out-of-domain 集，并报告词汇 overlap/TT ratio；不能随机混合朗读与自然会话。
- 若使用经文或朗读数据，先做无监督/表示继续预训练，再对目标会话微调；直接拼接会制造域偏置。
- 壮语/瑶语项目应把 speaker、录音设备、文体和文本来源纳入 split 元数据，按域报告 WER/CER。

## 是否值得精读
- 标签：值得精读
- 判断理由：对当前“文字重复/域迁移”问题给出跨语言证据和可执行两阶段训练方案。

