# 壮语与瑶语语音识别研究

项目根目录：`~/myfiles/asr_rssearch`（按用户指定拼写保留）。

## 当前资料

- [文献总目录](literature/README.md)：历史批次目录，附阅读顺序和用途。
- [迁移学习与快速成果路线](docs/迁移学习与快速成果路线.md)：2026-10-14方向比较、全数据实验矩阵与排期。
- [三个创新候选的详细实验设计](refine-logs/EXPERIMENT_PLAN.md)：跨说话人参数分配、音频证据蒸馏、选择性跨语迁移；含可能结果与结论边界。
- [PEFT近邻文献笔记](docs/literature/transfer-peft-directions.md)与[继续预训练/蒸馏笔记](docs/literature/transfer-distillation-cpt-notes.md)。
- [迁移适配补充参考库](literature/transfer-adaptation-references.bib)：独立书目，合并时按DOI/题名去重。
- [已完成5小时基线](docs/基线结果-2026-10-13.md)：工程试跑与共同开发集监控结果。
- [选题调研与实验建议](literature/选题调研与实验建议.md)：已有先例、候选研究问题及实验设计。
- [需要补充全文](literature/需要补充全文.md)：3篇尚未下载成功的论文，含题名、DOI与原因。
- [已下载文献BibTeX](literature/references.bib)。
- `tools/`：文献下载、目录生成和复制路径更新脚本，与下载的论文文件分开存放。

2026-10-12已从Downloads复制资料，并更新副本元数据中的绝对路径。19份PDF均与原始SHA256校验值一致；Downloads原件保留，没有覆盖既有研究文件。

数据检查、实验代码与结果以本目录为根目录。两条5小时工程基线已完成；完整训练池实验尚未启动。原始数据、转写和预测仅保留服务器，Git保存代码、书目与匿名报告。
