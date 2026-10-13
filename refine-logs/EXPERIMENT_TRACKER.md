# 点子一执行清单

日期：2026-10-14。用户已选定点子一。状态均为TODO，不代表已启动。

| ID | 任务 | 系统/数据 | 状态 | 决策依据 |
|---|---|---|---|---|
| P00 | medium缓存与LoRA参数覆盖核验 | 四组q/v，实际计数与公式一致，100–300step吞吐 | TODO | 显存/保存恢复/评分正确 |
| P01 | train内部三个speaker folds | split hash、时长、文本重合统计，仅train | TODO | 身份/内容覆盖清楚 |
| P02 | 12个首批探针 | 四组rank8×3fold；短预算 | TODO | heldout收益稳定、自由生成CER与loss排序相符 |
| P03 | 有界rank效用曲线 | 必要时补4/16，共最多36探针 | TODO | 实际参数成本及非加性检查 |
| B00 | 全数据强基线 | medium fullFT / 107h | TODO | 合理调参、完整dev |
| B01 | 参数分配对照 | uniform LoRA / AdaLoRA / 候选 / 同成本随机选择 | TODO | 预算公平，关键3seeds |
| A00 | 信号消融 | train gain / heldout gain / 稳定性项 | TODO | 独立机制贡献 |
| R00 | 条件性复核 | 20/50/107h关键方法，small、另一内部speaker分组 | TODO | 主结果成立再投入 |
| T00 | 冻结后终评 | 固定test矩阵，CER/token-WER/逐人 | TODO | 不反向调配方 |

先P00/P01/P02与B00并行准备。探针失效即停止扩大。所有原始数据/预测留服务器，Git保存脚本、匿名统计与报告。
