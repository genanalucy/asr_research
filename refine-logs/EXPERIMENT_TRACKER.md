# 三个创新候选实验跟踪表

日期：2026-10-14。所有行均为计划，不表示已运行。主线尚待机制小试筛选；计划细节与停止条件见 `EXPERIMENT_PLAN_2026-10-14_three-ideas.md`。Run ID表示实验组，真实启动后另记录每个seed、manifest hash和GPU。

| Run ID | 阶段 | 目的 | 配置/数据 | 优先级 | 状态 | 决策门 |
|---|---|---|---|---|---|---|
| C00 | 准备 | 数据/模型/吞吐校准 | 固定train/dev；medium/small输入与词表；100–300step真实测速 | 必须 | TODO | 确认预算、显存与评分正确 |
| C01 | 基线 | 共享强教师/迁移底座 | medium全FT / 107h | 必须 | TODO | 完整dev稳定，checkpoint固定 |
| C02 | 基线 | 共享学生/PEFT对照 | small全FT；medium uniform LoRA / 107h | 必须 | TODO | 合理配方，不使用弱调参对照 |
| I1-P | 点子一小试 | 模块效用可重复性 | 6组×内部3speaker folds，先一档rank；proxy | 必须 | TODO | loss排序与自由生成CER相符 |
| I1-M | 点子一主表 | 固定参数预算收益 | uniform/AdaLoRA/候选/等成本随机搜索；107h、3seed | 小试通过后必须 | TODO | 超出强基线与随机搜索 |
| I1-A | 点子一消融 | 区分信号/模块交互 | training-gain、speaker-gain、去稳定惩罚、联合模块检查 | 小试通过后必须 | TODO | 排除额外搜索/参数效应 |
| I1-R | 点子一复核 | 规模与分组稳健 | 20/50/107h关键方法、small、内部新speaker分组 | 主结果通过后必须 | TODO | 收益不局限单seed/单分组 |
| I2-P | 点子二诊断 | G是否超出teacher confidence | 内部heldout的matched/2wrong/静音；置信度分桶与频率/位置诊断 | 必须 | TODO | 不仅是置信或OOD脆弱性 |
| I2-M | 点子二主表 | 改善小学生KD | 无KD/标准KD/句CER筛选/token可靠度/联合G；107h、3seed | 小试通过后必须 | TODO | 超出已有筛选 |
| I2-A | 点子二消融 | 区分音频与KD强度 | G-only、可靠度-only、联合、随机等量w、匹配lambda/保留率 | 小试通过后必须 | TODO | G的独立增益 |
| I2-R | 点子二分析 | 错误继承与推理成本 | 逐人/错误重合/低频token/插删替；固定RTF协议 | 主结果通过后必须 | TODO | 不以删少见字符换低CER |
| I3-D | 点子三前置 | 供体可信 | 瑶语身份/来源/真转写/去重；保守划分 | 必须 | TODO | 未通过不得声称语言/speaker泛化 |
| I3-P | 点子三小试 | donor encoder/decoder干预 | 目标直接/供体全体/encoder-only/decoder-only；固定中等数据点 | 条件通过后必须 | TODO | 迁移效应是否存在 |
| I3-M | 点子三主表 | 更新筛选有效性 | full donor/encoder-all/selected/random/gradient norm；107h、3seed | 小试通过后必须 | TODO | 胜过encoder-only和随机 |
| I3-A | 点子三消融 | 排除额外预算 | 等额外target训练；供体/目标规模关键点 | 小试通过后必须 | TODO | 不是多训练产生的收益 |
| I3-R | 点子三验证 | 语言/来源外推 | 反向迁移或可靠公开第二供体；逐人/输出偏置 | 可选，普适主张时必须 | TODO | 单对语言case study边界 |
| FINAL | 终评 | 冻结方案后检验 | 幸存主线关键配置3seed、固定test矩阵 | 必须 | TODO | 不能用结果重新调参 |

优先共用C01/C02。本轮处于设计阶段，没有启动新训练。先机制诊断，再扩大训练，失败和实际时间写入跟踪表，不将TODO替换为未经运行的完成状态。
