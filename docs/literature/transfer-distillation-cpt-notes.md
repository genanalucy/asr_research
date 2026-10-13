# 迁移路线补充证据：继续预训练、蒸馏与文本解码

本轮目标：优先用完整训练样本提升ASR，同时确定可检验的方法改进。以下证据不证明任何路线在武鸣壮语/用户瑶语上必然有效，也不将论文数量当作查新完成。

## 继续预训练与多语言迁移

### Predicting positive transfer for improved low-resource speech recognition using acoustic pseudo-tokens

- 来源：https://arxiv.org/abs/2402.02302 ，已有本地全文13页与阅读卡。
- 本轮使用的证据：方法与附录预算，全文阅读记录见`literature/reading-notes/core/2402.02302_acoustic_transfer_selection.md`。
- 10h目标语音+60h供体做CPT，再用1h目标转写训练；每CPT约15小时、4×A100（附录A），并用声学伪token分布选择供体。
- 对本项目：可在已有标注训练音频上暂不用文本，先做目标语言自监督CPT；不必另有未标注数据，但这些样本不能被称为额外收集的未标注语料。需与直接监督微调、等额外更新/训练预算对照；不能拿test音频做CPT。
- 边界：原论文主要关注极少目标标签和供体筛选，本项目107h目标训练标签充足，额外CPT收益不确定；复现目标语CPT比复现整套供体排名更快。

### Evaluating the Effect of Linguistic Relatedness on Cross-Lingual Transfer in Large Multilingual Automatic Speech Recognition

- 来源：https://arxiv.org/abs/2607.04814 ，本地15页全文，方法/表格记录见已有精读卡；预印本发表状态未另核。
- 证据：顺序预适配再目标微调；在其条件下少量目标数据后相关/不相关供体差异很小。
- 对本项目：壮语(Tai-Kadai)与具体瑶语变体不能预设亲缘；壮语→瑶语迁移是候选实验，必须做直接瑶语适配、预算匹配和数据量控制。联合训练、顺序迁移和语言独立adapter不是同一种训练方式。
- 瑶语约37.83h候选仍需来源/身份核验；不能依据19个工作ID称19名独立说话人。

## Distil-Whisper：大教师、小学生

- 题名：*Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling*。
- 作者：Sanchit Gandhi、Patrick von Platen、Alexander M. Rush；2023 arXiv预印本，https://arxiv.org/abs/2311.00430。
- 下载：本地`literature/additions_transfer_20261013/distillation/2311.00430/paper.pdf`，30页，PDF签名、页数、SHA256已保存。
- 本轮读到：摘要、§3、§4.1–4.2、§7.1、§9.2–9.3、表1/2/12；不是完整30页精读声明。
- 方法：缩减层数的学生、序列伪标签与token概率KL结合；用真实参考与教师生成之间的WER阈值过滤伪标签（§4.2）。
- 原实验训练集21,170h、18,260+人、多个英语域（§7.1、表2）；公开蒸馏效果不能外推到壮语。
- 对本项目：先用完整107h训练强教师，再训练学生，保留真实转写硬监督，加教师软监督；如果没有新增无标签语音，不把已有人类标签简单替换为教师生成。
- 重要实现：teacher/student tokenizer及词表需对齐；large-v3与small的mel维度/语言token可能不同，不能直接盲比logits或复制不同宽度层。蒸馏方法本身已存在，贡献需要来自具体可靠性机制或验证新的准确率/速度边界。
- 验证：同学生、同数据、同更新预算的无KD/标准KD/候选机制对照，另报RTF、显存和参数量；教师错误放大是失败风险。

## 新主干迁移的实现证据

- 模型卡：https://huggingface.co/facebook/w2v-bert-2.0 ，本轮通过公开镜像读取官方仓库README。
- 官方声明：Conformer-based 600M encoder，4.5M小时未标注音频、143+语言预训练；bare encoder，没有ASR输出头，需fine-tune。
- 对本项目：作为XLS-R-300M之外的强SSL基线；使用自己的训练字表和CTC头，不能把模型卡的覆盖列表当作武鸣方言效果保证。
- 该checkpoint使用声学filterbank输入，与已有XLS-R原波形管线不同，不能只替换模型名而不修改processor。

## 文本与解码补充

- *Whisper-LM: Improving ASR Models with Language Models for Low-Resource Languages*，https://arxiv.org/abs/2503.23542 ，项目已有26页PDF；本轮仅使用既有摘要/目录记录，没有宣称新读完。
- *Mai Ho'omāuna i ka 'Ai: Language Models Improve Automatic Speech Recognition in Hawaiian*，https://arxiv.org/abs/2404.03073 ，已有5页PDF；外部LM重打分有先例。
- 可快速用训练集文字训练n-gram/轻量LM，比较greedy/beam/LM rescoring；开发/测试参考不得进入LM训练。已有配对训练文字可用，但如果声称独立外部文本，需要真实额外来源。
- LLM自由纠错可能改变音频未说的内容；优先受声学分数、n-best及编辑范围约束的方案。普通LM重打分不是首创；当前不把7B文本LLM作为第一阶段依赖。

## 调参直接近邻与最新适配近邻

- DOI `10.3390/app152413090`：*Low-Resource Speech Recognition by Fine-Tuning Whisper with Optuna-LoRA*。Crossref题名、作者和出版条目及PDF首页摘要已核；最后取得22页全文，但实验正文未读，不能据此报告经正文核验的指标。已足以提醒Optuna+LoRA组合不是未出现过的方向。
- https://arxiv.org/abs/2609.29800 ：*Adaptive Fisher-Whitened Cross-Covariance for Low-Resource Speech Recognition*。本轮只读官方页面摘要；涉及task-informed子空间、cross-layer sharing和固定参数预算的adaptive rank。最后取得5页全文，正文尚未核读。
- 因此自动rank分配、跨层参数共享等需要和AdaLoRA/FCCA类近邻对照，不能改名后直接称首创。
