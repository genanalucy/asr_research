# 三个候选查新运行记录

日期：2026-10-14，北京时间。当前用户请求：选择三个相对有创新的ASR点子，分别详析实验、可能结果与最终可支持结论。

## 执行范围

- 本轮首次创建新Luna子代理失败：`agent thread limit reached`。
- 因运行时代理线程限制，复用 `/root/transfer_peft_research` 做公开近邻检索，沿用其Luna设置；复用 `/root/assess_refined_innovation` 做方法咨询，其实际模型身份本次工具输出未提供，不做模型身份推断。
- 未达到novelty-check的fresh GPT-6-Astra reviewer要求；咨询复核保留前文共享上下文，不称独立/盲审或正式查新通过。
- novelty-verdict-limits完整文本已发给咨询代理。查询与公开来源限制由实际检索稿记录，不称穷尽查新。
- 未启动训练、未读取test、未发送真实音频/转写/姓名给公开检索。

## 原始候选与主张

1. train内部speaker-heldout适配效用/稳定性 → 固定LoRA参数预算分配；与uniform/AdaLoRA/训练信号及同成本搜索区分。
2. 同gold前缀matched与错配train音频的teacher真实token logprob差 → KD权重；与confidence/correctness、参考错误筛选及同有效KD量区分。
3. donor encoder更新在target-train梯度下的兼容性 → 部分模块迁移，decoder从base开始；与encoder-all、随机筛选及等额外target-only预算区分。

## 已收到的原始咨询结论摘要

代理 `/root/assess_refined_innovation` 返回：

> 1 PROCEED WITH CAUTION：真实delta是train内heldout-speaker实际收益+稳定性，注意probe公平搜索和nonadditive模块。
> 2 PROCEED通过本次早期近邻门：wrong-audio同goldprefix gate有可辨认delta，需强confidence+correctness+matched retention/random对照；分数可能只是incongruence而非因果grounding。
> 3 PROCEED WITH CAUTION，创新最薄：LoRA donor warm start已有2024/2026；encoder-only/reset decoder和target effective-gradient筛选需补专项查新，语言/来源风险待解。

完整咨询输出保存在 `docs/literature/three-ideas-method-review-2026-10-14.md`，检索输出在 `docs/literature/three-ideas-prior-search-2026-10-14.md`。原咨询引用S2-LoRA的简称题名已由主代理修正为官方题名，arXiv ID未变。

## 主代理最终处置

- 所有候选允许小试；不将任一候选标成“首创已核验”。没有找到包含完整候选与结果的指定论文，因此未作ABANDON。
- 方法贡献争取空间：2 > 1 > 3；最快真实对照：1 > 2 > 3。均为定性研究判断。
- 纳入意见：真实参数成本、同搜索预算、单模块非加性、token loss与自由生成CER区别、same/novel内容分层、错配分布外、G与confidence重合、KD平均强度、special/pad exclusion、effective权重更新梯度而非LoRA因子、供体身份/域混杂。
- 执行稿采用6组粗模块，最多36探针；咨询稿8组24探针为未采纳的候选设置，非实际运行数。
- 对核心接近来源作官方arXiv元数据核验；新保存3份已下载PDF和5条元数据/书目。具体阅读范围逐篇保留。
- 主方案/跟踪表所有实测状态均为未运行，可能结果均为条件分支。
