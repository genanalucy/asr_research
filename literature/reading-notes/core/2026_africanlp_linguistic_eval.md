# 论文精读卡

## 输入覆盖范围
- 来源：[paper.pdf](../../2026_africanlp_linguistic_eval/paper.pdf)，AfricaNLP 2026 全文。
- 覆盖级别：全文（PDF 14 页；已读摘要、§1--§8、表 1--11、附录 A--C）。
- 边界说明：条目为未来年份会议论文，正式元数据状态未在本次精读中另核；结论只按 PDF 实验报告。

## 标题
Linguistically Informed Evaluation of Multilingual ASR for African Languages

## 链接
https://aclanthology.org/2026.africanlp-main.14/

## 研究问题
- WER/CER 是否掩盖低资源、声调语言的部分音系学习；能否用 PanPhon 风格 Feature Error Rate（FER）和 Tone Error Rate（TER）补充评价（PDF pp.1--3）。
- 另测自然会话与慢速逐词 careful speech 的域移位，使用 Yoruba（预训练见过）与 Uneme（未见过）及 English 对照。

## 核心方法
- 冻结 mHuBERT-25-Hz、mHuBERT-147 或 XLS-R encoder，仅训练 Linear 或两层 Transformer CTC decoder（PDF pp.4--5，表 7）。
- G2P 后把每个 segment 映射到 24 维 {-1,0,+1} 音系特征；对音调维度单独算 TER；用 NA-masked Needleman-Wunsch 对齐后求 FER（PDF pp.5--6）。

## 模型 / 算法要点
- Uneme 约 8 h、7 speakers，4,729 train/880 val/1,027 natural test/1,051 careful test；careful speech 仅测试，不进训练/验证（表 2，PDF p.4）。Yoruba/English 使用 FLEURS，分别 3,548/3,643 utterances（表 1）。
- 最佳 decoder 是两层 Transformer；训练 20,000 steps、lr 5e-5、batch 16、grad accumulation 4、effective batch 64、1,000 warmup（附录表 7，PDF p.10）。

## 实验设置
- 指标：WER、CER、FER、TER 及按特征/声调类别 FER；错误示例做定性分析（表 3--6）。
- baseline：mHuBERT-25-Hz、mHuBERT-147、XLS-R；Linear vs Transformer。

## 主要结果
- Yoruba 最佳 mHuBERT-147 + Transformer：WER .788、CER .305、FER .151、TER .372；English 同设置 WER .511、CER .167、FER .144（表 3，PDF p.7）。
- Uneme natural 上最佳 mHuBERT-147 + Transformer：WER .997、CER .461、FER .267、TER .402；careful 上 WER 1.622、CER .837、FER .515、TER .510（表 4--5，PDF pp.7--8）。
- Uneme 下 back-vowel FER .280、downstep FER .565，说明词级失败中仍保留大量音系信息（表 4--5，PDF pp.7--8）。
- 论文将 careful speech 变差归因于自然会话训练到逐词朗读的 prosody/domain shift，并承认没有显式建模 speaker age/gender/accent（§6.4、§7，PDF pp.9--10）。

## 优势
- 将 tone 和 articulatory features 纳入可计算指标，能区分“词错但音系接近”和真正音段错。
- 数据设有 natural/careful 两个 test，且 careful 不泄漏训练，适合检验域移位。

## 局限
- FER 高度依赖 G2P；深层正字法或非 IPA 书写不适用（§7，PDF p.10）。
- Uneme 数据规模和说话人仍小，句子按停顿自动切段，可能混入非语言切分误差。
- encoder 冻结、只训 decoder 的设置适合诊断表示，不等同于完整微调 ASR；XLS-R 线性头几乎退化。

## 复现难点
- 需实现 24 维语言特定 feature map、tone mapping 和 NA-masked edit distance；附录表 8--11 才给全量映射。
- G2P 和音调标注要有语言专家；仅依赖字符 CER 无法复现 FER/TER。

## 对当前研究的启发
- 壮语/瑶语至少报告 WER、CER，并设计声调/韵母/声母特征 FER；若无法可靠 G2P，应明确标注“未实现 FER”，不伪造音系结论。
- 测试集加入自然会话与朗读/逐词条件，保留风格标签；careful speech 不能自动当作更容易的测试域。
- 先用冻结 encoder + 轻量 decoder 做表示诊断，再做 full fine-tune，避免把预训练覆盖和音系能力混在一起。

## 是否值得精读
- 标签：值得精读
- 判断理由：提供当前项目所需的音系评测定义、tone 误差分解与域移位对照；但未来年份条目的出版状态需另核。

