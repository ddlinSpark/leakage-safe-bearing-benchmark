# 文档索引

| 文档 | 内容 |
| --- | --- |
| [../REPRODUCE.md](../REPRODUCE.md) | 从零复现数据、训练、统计、图表和论文数字的完整步骤 |
| [PUBLIC_BENCHMARK.md](PUBLIC_BENCHMARK.md) | 当前公共数据主实验、五种子设计和泄漏安全划分 |
| [DATASET.md](DATASET.md) | 官方数据来源、选择轴承、窗口、缓存与 damage-origin 映射 |
| [PROTOCOL.md](PROTOCOL.md) | fold 定义、泄漏边界、预算与 fold-first 统计协议 |
| [EXPERIMENTS.md](EXPERIMENTS.md) | 正式实验、敏感性分析、外部适配与运行状态 |
| [RESULTS.md](RESULTS.md) | 主结果、泛化差距、自训练、校准和图表数据来源 |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | 统一 runner、配置快照、统计和图表复现 |
| [SCIENTIFIC_REVISION.md](SCIENTIFIC_REVISION.md) | 原稿方法问题、修订问题与报告规范 |
| [paper/PAPER_REPRODUCIBILITY.md](paper/PAPER_REPRODUCIBILITY.md) | 论文构建与数值验证流程 |
| [paper/PAPER_CODE_MAPPING.md](paper/PAPER_CODE_MAPPING.md) | 论文方法、RQ、代码、配置、结果和图表映射 |
| [paper/FIGURE_PROVENANCE.md](paper/FIGURE_PROVENANCE.md) | Figure 1-12 与生成脚本及输入数据 |
| [../figures/publication/figure_hashes.json](../figures/publication/figure_hashes.json) | 全部 publication 图件的 SHA-256 与字节大小清单 |
| [../results/README.md](../results/README.md) | canonical、external、sensitivity 与 post-hoc 结果分类索引 |
| [../results/canonical/manifest.json](../results/canonical/manifest.json) | 正式论文结果的路径、行数、大小与 SHA-256 清单 |
| [../splits/manifest.json](../splits/manifest.json) | 四个冻结 fold manifest 的 SHA-256 索引 |
| [../CONSISTENCY_REPORT.md](../CONSISTENCY_REPORT.md) | 自动检查 run 数量、模型、指标、区间、参数、fold、seed 与 DOCX 一致性 |
| [../CLEAN_REPRODUCTION_REPORT.md](../CLEAN_REPRODUCTION_REPORT.md) | 从公开 GitHub 新克隆后的预处理、训练、统计、图件、DOCX 与 PDF 复现记录 |
| [../FINAL_RELEASE_REPORT.md](../FINAL_RELEASE_REPORT.md) | 最终发布变更、图件、配色、一致性、目录、排除文件和剩余警告 |
| [paper/TABLE_PROVENANCE.md](paper/TABLE_PROVENANCE.md) | Table 1-13 与生成脚本及聚合规则 |
| [audit/REPOSITORY_AUDIT.md](audit/REPOSITORY_AUDIT.md) | 仓库结构、README、依赖和数据许可审计 |
| [audit/LEAKAGE_FINAL_CHECK.md](audit/LEAKAGE_FINAL_CHECK.md) | 最新 split、训练、伪标签和校准泄漏检查 |
| [audit/RESULT_COMPLETENESS_AUDIT.md](audit/RESULT_COMPLETENESS_AUDIT.md) | 320 条 canonical 与 40 条 external 结果完整性 |
| [audit/RELEASE_READINESS.md](audit/RELEASE_READINESS.md) | v1.0.0 发布准备度、阻断项和严重性 |
| [audit/GIT_HISTORY_PRIVACY_AUDIT.md](audit/GIT_HISTORY_PRIVACY_AUDIT.md) | 全 Git 历史的隐私与敏感信息审计 |
| [audit/DOUBLE_BLIND_RISK.md](audit/DOUBLE_BLIND_RISK.md) | 双盲投稿身份暴露风险与可选处理方式 |
| [audit/ZENODO_PREPARATION.md](audit/ZENODO_PREPARATION.md) | GitHub Release 后 Zenodo 归档步骤 |
| [../results/EXPERIMENT_REPORT.md](../results/EXPERIMENT_REPORT.md) | 当前五-Seed 实验结果、区间与配对检验 |

论文正文位于 `docs/paper/`，参考文献位于 `docs/references/`。
投稿 PDF 位于 `manuscript/`，全部图件统一使用
`src/spindle_cnn/visualization/style.py` 中的 palette。
