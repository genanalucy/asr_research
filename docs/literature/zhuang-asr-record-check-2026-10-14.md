# 壮语公开 ASR WER 纪录核验

检索日期：2026-10-14。用户所说的“最高纪录”按最好成绩理解，即最低错误率。

## 结论

**本次检索未能核实一个可比的、公开发表的壮语原文连续语音识别（ASR）最低 WER。**因此不能负责任地给出“当前壮语 WER SOTA = 某百分比”，也不能把未找到纪录理解成不存在相关系统。

检索到的直接壮语候选论文中，一篇是2011年题名含“speech recognition”的会议论文，但可访问题录没有摘要或指标，全文请求失败；另一篇2026年论文是壮语到中文的语音翻译（S2TT），不是输出壮文转写的 ASR，且本轮未取得全文。当前没有证据证明2011论文做的是连续词级识别，也没有证据能从它得出 WER。

## 检索范围

使用了至少四组英文/中文查询：

- `Zhuang ASR word error rate continuous speech`
- `壮语 语音识别 词错误率 连续语音`
- `Northern Wuming Zhuang Whisper XLS-R ASR`
- `Zhuang automatic speech recognition benchmark`
- 补充精确题名查询：`"An new speech recognition method based on prosodic analysis and SVM in Zhuang language"`

检索来源包括 arXiv API、Crossref DOI 题录、IEEE Xplore/ACM 官方入口，以及公开网页搜索。arXiv 查询主要返回同名作者或无关 ASR 项；网页搜索的一次中文结果只命中通用 WER/CER 说明，未找到壮语基准数字。没有用私人数据、测试集或用户语音。

## 候选记录与可确认内容

| 记录 | 原文位置和已核范围 | 对 WER 纪录的意义 |
|---|---|---|
| Yin & Tian (2011), [*An new speech recognition method based on prosodic analysis and SVM in Zhuang language*](https://doi.org/10.1109/MEC.2011.6025684), DOI 10.1109/MEC.2011.6025684 | Crossref 题录：2011 IEEE MEC，pp.1209–1212，作者 Yeqing Yin、Tao Tian；IEEE Xplore 文档号 [6025684](https://ieeexplore.ieee.org/document/6025684)。Crossref 没有 abstract。IEEE 官方 PDF 请求返回 HTTP 418 HTML；同一 PDF 的仓库登记也记录 HTTP 418。PDF 未取得，未读实验。 | 题名能确认论文与壮语语音识别有关；但任务单位、语料/方言、划分和指标均未核实。不能将其当成连续 ASR 的 WER 基线，更不能从题名推断识别率。手动获取入口：[IEEE Xplore 文档页](https://ieeexplore.ieee.org/document/6025684)。 |
| Nong, Chen & Zhang (2026), [*Zhuang-CN S2TT: A Low-Resource Dataset for Zhuang End-to-End Speech-to-Text Translation*](https://doi.org/10.1145/3838457.3838547), DOI 10.1145/3838457.3838547 | Crossref 官方题录：2026-05-15 出版，pp.607–612，ACM；题名明确为 speech-to-text translation。Crossref abstract 为空。ACM PDF 入口返回 HTTP 403 HTML；未取得或阅读全文。 | 任务是壮语语音到中文文本，目标输出不是壮语原文转写。即使原文报告 BLEU、翻译准确率或其他译文指标，也不能冒充壮语 ASR WER。手动获取入口：[ACM Digital Library PDF](https://dl.acm.org/doi/pdf/10.1145/3838457.3838547)。 |

以上是本轮直接壮语候选的题录核验，不是断言这两篇构成完整文献集。当前本地文献目录也把2011论文列为待补全文，把2026 S2TT 标为翻译任务；本次没有修改它们的元数据或目录。

## 指标不可混用

- **连续原文 ASR WER**：必须有逐句/连续语音参考转写和识别转写，并按明确的词/空格单位计算 `WER=(S+D+I)/N`。壮文的正字法分词规范会影响结果，跨论文比较前需统一单位和文本规范化。
- **CER**：按字符编辑距离计算，不等于 WER；没有正字法 tokenization 规则时，CER 可能更可重复，但不能改名为 WER。
- **孤立词、音节或音节/声调分类准确率**：单位、候选集合和评分机制不同于句级 ASR WER，不能直接排名。
- **壮语到中文 S2TT**：目标文本是中文译文，常用翻译指标；不能据此报告壮文识别错误率。

## 对当前成绩说法的建议

可准确表述为：“截至2026-10-14，本次公开检索未确认有可比的连续壮语原文 ASR WER 纪录。已核实的壮语相关公开工作包括一篇未取得全文/摘要的2011题录，以及一篇2026壮语到中文语音翻译数据集论文；两者均不能提供本轮可核验的 ASR WER。”不应写“全球最好”“打破 SOTA”或填入未经原文核实的百分比。

如果后续取得2011论文全文，应先核实其输出是词、音节、声调类别还是句级转写，再检查测试集、分词与评分公式。若发现确有可比 WER，还需对齐方言/录音域、说话人划分、训练和测试数据、识别目标及文本规范后，才能称其为公开基线。
