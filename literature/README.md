# 壮语与瑶语ASR论文资料包

整理日期：2026-10-12。

- 已下载 **29篇不同论文、30份PDF**。Whisper公开预印本与ICML正式版属于同一篇，不重复计数；总目录优先指向ICML正式版。
- **3篇尚未取得全文**：见[需要补充全文](需要补充全文.md)。
- 所有30份PDF均检查了文件签名、可解析页数、首页题名和SHA256；这不等于已经逐篇精读全文。
- 每篇PDF单独目录，附README、元数据和首页文本。历史失败目录保留供追溯；有成功重试的论文不列为待补。
- [选题调研与实验建议](选题调研与实验建议.md)：候选方向、创新边界、实验设计和所需数据条件。
- [references.bib](references.bib)：已取得全文的论文，可导入文献管理软件；正式出版信息请投稿前核对。
- [pending-fulltext.bib](pending-fulltext.bib)：尚未取得全文的三篇，单独保存，不冒充已读文献。
- [SHA256SUMS.txt](SHA256SUMS.txt)：文件校验值。

## 建议阅读顺序与目录

A表示选题/基线/近邻查新优先；B表示补充背景或特定路线的参考，不代表论文质量高低。以下用途是调研判断，不是原论文结论的替代。

| 阅读顺序 | 题名 | 优先级 | 全文 | 用途 |
|---|---|---|---|---|
| 1 | Automatic Speech Recognition for Documenting Endangered Languages: Case Study of Ikema Miyakoan | A | [PDF](2026_lrec_ikema/paper.pdf) · 9页 | 多说话人濒危语言、正字法比较和ASR辅助标注效率；用于设计外推与实用评测。 |
| 2 | Linguistically Informed Evaluation of Multilingual ASR for African Languages | A | [PDF](2026_africanlp_linguistic_eval/paper.pdf) · 14页 | FER/Tone Error Rate与WER/CER并列的核心评测近邻；适合构造壮语/瑶语音系错误分析。 |
| 3 | ManWav: The First Manchu ASR Model | A | [PDF](2024_fieldmatters_manchu/paper.pdf) · 6页 | 极低资源ASR、音系错误和增强实验的参考；明确报告说话人限制。 |
| 4 | In-context Language Learning for Endangered Languages in Speech Recognition | A | [PDF](2025_in_context_endangered_asr/paper.pdf) · 6页 | MMS n-best与目标语文本重排序；若走外部LM/文本适配路线，必须做公平基线。 |
| 5 | Phoneme transcription of endangered languages: A comparative study of architectures and data regimes | A | [PDF](2022_findings_phoneme_endangered/paper.pdf) · 8页 | 多种濒危语言、短时长和PER/架构比较；支持音素级错误诊断与数据规模实验。 |
| 6 | Enhance Contextual Learning in ASR for Endangered Low-resource Languages | A | [PDF](2025_lm4uc_endangered_context/paper.pdf) · 7页 | 低资源语言tokenization、上下文学习和跨语言参数迁移；若研究解码/文本适配可作近邻。 |
| 7 | Tusom2021: A Phonetically Transcribed Speech Dataset from an Endangered Language for Universal Phone Recognition Experiments | A | [PDF](2021_interspeech_tusom/paper.pdf) · 5页 | 濒危语言音素标注、跨语言phone recognition和声调背景参考。 |
| 8 | Faithful Transcription: Leveraging Bible Recordings for Low-Resource ASR | A | [PDF](2025_ijcnlp_faithful_transcription/paper.pdf) · 10页 | 朗读域与自然会话域差异；支持域外测试和语料风格控制。 |
| 9 | Leveraging End-to-End ASR for Endangered Language Documentation: An Empirical Study on Yoloxochitl Mixtec | A | [PDF](2021_eacl_mixtec/paper.pdf) · 12页 | 约100小时濒危语言ASR和数据规模曲线先例；参考speaker-disjoint、10/20/50/90小时扩展，不宣称方法首创。 |
| 10 | Improving ASR Output for Endangered Language Documentation | A | [PDF](2018_sltu_endangered_doc/paper.pdf) · 5页 | ASR输出后处理、词典与语言文档工作流背景；适合讨论实用价值。 |
| 11 | NoLoR: An ASR-Based Framework for Expedited Endangered Language Documentation with Neo-Aramaic as a Case Study | A | [PDF](2024_nolor_neo_aramaic/paper.pdf) · 6页 | ASR辅助语言文档与人机协作效率；若讨论实用影响可引用，不作为主模型近邻。 |
| 12 | Low-Resourced Speech Recognition for Iu Mien Language via Weakly-Supervised Phoneme-based Multilingual Pre-training | A | [PDF](2407.13292_iu_mien_asr_retry/paper.pdf) · 5页 | 最直接的勉语ASR先例；MightLJSpeech约9.7小时、9761句、单女性说话人。此次按原页面的Pre-training拼写修正题名后下载。 |
| 13 | Whistle: Data-Efficient Multilingual and Crosslingual Speech Recognition via Weakly Phonetic Supervision | A | [PDF](2406.02166_whistle/paper.pdf) · 14页 | 弱音素监督、IPA桥接与多语言预训练的核心先例；勉语论文使用这条路线。 |
| 14 | SITA: Learning Speaker-Invariant and Tone-Aware Speech Representations for Low-Resource Tonal Languages | A | [PDF](2601.09050_sita/paper.pdf) · 20页 | 说话人不变与声调敏感表示学习的直接近邻；同词异调的tone-repulsive loss已有先例。 |
| 15 | Error Rate is Not Enough: Costs of Augmentation in Low-Resource ASR | A | [PDF](2610.13807_augmentation_error_types/paper.pdf) · 5页 | 数据增强与细粒度错误分析；声调结果仅来自Kakabe，单独SpecAugment的声调错误增加不显著，不可推广为壮瑶语必然现象。 |
| 16 | Phoneme-Aware Augmentation for Robust Cantonese ASR Under Low-Resource Conditions | A | 待补全文 | phoneme-aware SpecAugment与边界感知增强的直接近邻；新方法必须与它区别，不能把边界内mask或轮廓保护单独作为新意。 |
| 17 | T-SANDHI: Tone Sandhi-aware Adaptive Network with Decoupled Hybrid Injection for Low-resource Taiwanese Hokkien Speech Recognition | A | [PDF](2609.18194_tone_sandhi_retry/paper.pdf) · 7页 | 声调/音节辅助任务、动态门控和本调/变调解耦的直接近邻。首次下载遭遇临时TLS错误，此为独立目录的重试。 |
| 18 | Explicit Tone Transcription Improves ASR Performance in Extremely Low-Resource Languages: A Case Study in Bribri | A | [PDF](explicit_tone_bribri_explicit_tone_bribri/paper.pdf) · 12页 | 显式声调转写/将声调与元音分开建模的早期先例；不能把声调分解本身宣称首创。 |
| 19 | Learn and Don't Forget: Adding a New Language to ASR Foundation Models | A | [PDF](2407.06800_new_language_adaptation/paper.pdf) · 6页 | 新增语言适配、soft language code、soft prompt、LoRA、全量微调与遗忘权衡；普通加语言token或LoRA不是新方法。 |
| 20 | Scaling Speech Technology to 1,000+ Languages | A | [PDF](2305.13516_mms/paper.pdf) · 41页 | MMS大规模语言覆盖与低资源ASR基线；语言是否覆盖需查官方语种清单，不能由民族名称推断。 |
| 21 | Robust Speech Recognition via Large-Scale Weak Supervision | A | [PDF](whisper_icml2023_whisper_icml2023/paper.pdf) · 27页 | Whisper优先引用版本。与2212.04356的arXiv下载属于同一篇论文，不应作为两篇文献重复计数。 |
| 22 | XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale | A | [PDF](2111.09296_xls_r/paper.pdf) · 23页 | 跨语言自监督语音表示与CTC微调基线；比较预训练主干而非只用Whisper。 |
| 23 | wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations | B | [PDF](2006.11477_wav2vec2/paper.pdf) · 19页 | 自监督预训练、低标注量ASR和CTC训练的基础方法文献。 |
| 24 | Tone in Yoruba ASR: Evaluating the Impact of Tone Recognition on Transformer-Based ASR Models | B | [PDF](tone_yoruba_tone_yoruba/paper.pdf) · 8页 | 声调标记、正字法和评价指标的边界案例；不能假设显式标调必然改善ASR，更不能直接外推到壮语/瑶语。 |
| 25 | Predicting positive transfer for improved low-resource speech recognition using acoustic pseudo-tokens | B | [PDF](2402.02302_acoustic_transfer_selection/paper.pdf) · 13页 | 基于声学伪token相似度选择迁移供体的ATDS先例；语言亲缘关系不是唯一选择标准。 |
| 26 | Evaluating the Effect of Linguistic Relatedness on Cross-Lingual Transfer in Large Multilingual Automatic Speech Recognition | B | [PDF](2607.04814_related_language_transfer/paper.pdf) · 15页 | 跨语言迁移的反证与实验设计参考；结论针对辅助语顺序预适配，不是否定多语预训练或联合训练。 |
| 27 | Tone-Conditioned Curriculum Learning for Low-Resource Bantu Speech Recognition | B | [PDF](2606.31642_tone_conditioned_curriculum/paper.pdf) · 5页 | 声调统计、课程学习和适配器的近邻；实际用整句F0统计而非人工词汇声调标签，没有独立TER。 |
| 28 | Hard to Be Heard: Phoneme-Level ASR Analysis of Phonologically Complex, Low-Resource Endangered Languages | B | [PDF](2604.18204_phoneme_level_analysis/paper.pdf) · 15页 | 低资源语音识别的音素错误分析与训练频次机制；不能仅把少见语言或音系复杂性当作方法创新。 |
| 29 | Mai Ho'omāuna i ka 'Ai: Language Models Improve Automatic Speech Recognition in Hawaiian | B | [PDF](2404.03073_hawaiian_lm_retry/paper.pdf) · 5页 | 低资源语言外部LM重打分先例。原页面题名含LaTeX形式的长音符，此次先解码为ā再核对，并非另一篇论文。 |
| 30 | Whisper-LM: Improving ASR Models with Language Models for Low-Resource Languages | B | [PDF](2503.23542_whisper_lm/paper.pdf) · 26页 | 统计LM与LLM辅助Whisper的低资源识别比较；用于评估是否真的有必要用大语言模型。 |
| 31 | An new speech recognition method based on prosodic analysis and SVM in Zhuang language | A | 待补全文 | 壮语ASR与韵律方法的已有先例；题名中的An new按原记录保留。不报告尚未核实的模型细节或性能。 |
| 32 | Zhuang-CN S2TT: A Low-Resource Dataset for Zhuang End-to-End Speech-to-Text Translation | B | 待补全文 | 壮语数据采集、切分和对齐验证参考；任务是语音到中文翻译，不是壮语原文ASR，不能直接混作ASR基准。 |

## 引用与使用注意

1. 下载的arXiv版本与正式发表版本不混同。已录用不等于已经核实最终页码、卷期。BibTeX按实际获取版本生成，不编造会议出版细节；Whisper ICML条目已核对PMLR官方页面。
2. 声调辅助、音节分解、语言token、LoRA、LM重打分已有先例；换到壮语/瑶语不自动构成方法创新。声调保护增强也须精读粤语近邻全文后再定差异。
3. 2026年Error Rate is Not Enough的声调结果来自Kakabe，不能直接外推到壮语/瑶语；相关亲缘迁移研究的结论是针对辅助语顺序预适配，不是否定所有多语言训练。
4. 壮语—中文S2TT是翻译任务，不能把中文译文混作壮语原文ASR转写。Iu Mien不代表所有被称为“瑶语”的变体。
5. 参考文献中部分新作仅为预印本；收录不代表其结果已经独立复现。

## 资源与官方入口（不是额外论文）

- MightLJSpeech勉语数据：https://github.com/mightmay/MightLJSpeech （单说话人朗读；用户是否同语种仍需确认）
- Whistle/CAT实现入口：https://github.com/thu-spmi/CAT/tree/master/egs/cv-lang10
- MMS官方说明与语种清单：https://github.com/facebookresearch/fairseq/tree/main/examples/mms
- Whisper官方实现：https://github.com/openai/whisper
- XLS-R/wav2vec实现：https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec

本次未下载数据集、运行模型训练、上传用户数据或改动当前插件项目。
