# 当前远程计算环境与复用证据

更新：2026-10-14，北京时间。后端：SSH，`hzh@10.240.147.134`，项目`/home/hzh/asr_research`，运行时`.venv/bin/python`。本轮复用既有环境，未安装/升级包。

## 声明配置与当前观测

- Python3.12.7，torch2.6.0+cu124，torchaudio2.6.0+cu124，transformers4.46.3，numpy2.5.4，soundfile0.12.1。
- 服务器8张L20、每张约46GiB。启动前只有GPU5/6/7空闲；0–4已有其他用户任务。每次实际启动还须nvidia-smi重新检查。
- 离线训练：HF_HUB_OFFLINE=1、TRANSFORMERS_OFFLINE=1；模型下载阶段使用已核对revision的公共镜像，不向其发送数据。
- 模型与数据本地化；WandB未配置，使用JSONL/日志。真转写/预测/音频/权重留服务器。

## 实际验收

本轮在空闲GPU5执行seed42、64×64 BF16矩阵乘法：结果finite，float32聚合sum4275.62890625；实际CUDA/BF16支持通过。随后small新代码真实M0包含16条训练音频过拟合、model/optimizer/scheduler恢复，gate全部通过。只导入包不作为就绪证明。

full缓存：59,925条全部波形与16k mono FLOAT缓存检查通过，0失败。small模型audit已校验哈希与CPU加载；medium也已完成全部文件SHA与离线CPU实际模型/processor加载，status为`offline_cpu_load_verified`。验收没有运行medium GPU forward，实际GPU训练可运行性由每条M0继续验证。

复杂代码复用既有审查代理进行咨询检查；不是fresh跨模型独立审计。训练执行代码固化到各run/code并记录SHA，避免后续编辑改变运行中实现。实际invocation与门控见`docs/全数据启动审查-2026-10-14.md`及启动记录。
