# 瑶语来源复核匿名汇总

日期：2026-10-13。此页只保留匿名统计；完整音频哈希、原始标签、来源路径和候选来源别名仅保存在服务器私有审计目录。

## 身份与重复音频

- 清单共 33,000 条已配对记录，29,855 个不同的完整 SHA-256 音频哈希。
- 3,113 个完整哈希重复组都只能作为来源别名或重复记录的候选链接；其中 2,326 组跨 TXT/XLSX 来源。不能据此确认独立说话人，也不能把未匹配的候选 ID 当作已确认不同的人。
- 现有候选身份数量是来源字段的工作分类，不是经过人工或声学核验的说话人数。

## 977 组标签冲突

按现有复核文件使用的 NFC/空白归一化标签分组，977 组分为：

- 实质字词或片段差异：894 组
- 标点差异：69 组
- 数字差异：8 组
- 大小写/空白归一化后仍有的格式差异：5 组
- 格式与数字同时变化：1 组

纯格式类别只提出人工复核和统一规范化建议，不自动改标签；实质差异保留为冲突，不能自动选边。

## 旧 TXT test 提示与泄漏

- 旧 TXT 路径提示有 49 条，49 个哈希均未在当前清单中形成重复音频组，因此没有发现这些提示对应的直接跨来源音频哈希重合。
- 整体数据仍有 2,326 个跨 TXT/XLSX 的完整哈希重复组。冻结 train/dev/test 时必须按完整哈希组整体留出，并重新核查来源录音和文本重叠；旧路径中的 `test` 文字不能直接作为有效评测切分。

## 远端只读环境

- Python：`/usr/bin/python3`，3.8.10；未发现 conda 命令或环境列表。
- `torch`、`transformers`、`soundfile`、`numpy`、`datasets`、`accelerate` 当前均不可导入；ffmpeg 为 4.2.7。
- 8 张 NVIDIA L20，每张 46,068 MiB，驱动 580.126.09。
- `/home/hzh/asr_research` 所在文件系统约 8.0 TB，总用量 5.6 TB，可用 2.1 TB（74%）；`/home/hzh/.cache/huggingface` 已存在但约 89.6 KB。
- `huggingface.co:443` 当前连接被拒绝，未尝试下载模型；PyPI 可达。因此当前模型下载不可行，需先恢复 Hugging Face 网络或准备经批准的离线模型缓存，再另行配置环境。

详细私有结果：服务器 `/home/hzh/asr_research/audit/private/yao_followup/` 下的 `report.md`、`review_stats.json`、`identity_hash_links_private.json`、`label_conflict_categories_private.json`、`txt_test_leakage_private.json` 和 `environment_report.md`。
