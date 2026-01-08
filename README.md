# CFPS 农业AI与TFP 实证分析（可复现流程）

本仓库用于记录与复现实证分析全过程：从 CFPS 数据挑选、变量扫描、TFP 计算、AI_Index（PCA）构建，到 OLS/PSM/因果森林替代估计与论文可替换输出。

## 合规与重要声明（必须阅读）

- CFPS 数据受使用协议约束，**严禁上传/公开传播原始数据文件（.dta）**。
- 本仓库默认通过 `.gitignore` 忽略：
  - `原始数据/`、`data/`
  - 记录级输出：`output/processed_data*.csv`
- 如需公开展示结果，建议仅上传汇总级输出（例如描述性统计、回归结果表、PCA载荷、方法对比图）。

## 目录结构

- `causal_analysis.py`：主脚本（生成论文完整口径的结果）
- `scan_vars.py` / `make_mapping.py` / `find_*_vars.py`：变量扫描与定位脚本
- `全过程记录.md`：全过程记录文档（可作为附录/过程材料）
- `output/`：结果输出目录（部分文件被 gitignore）

## 环境要求

- Windows x64
- Python 3.14（当前项目使用）

## 安装依赖

在项目根目录执行（PowerShell）：

```powershell
python -m pip install --upgrade pip
python -m pip install pandas numpy pyreadstat
python -m pip install scikit-learn statsmodels matplotlib seaborn
```

## 一键生成“论文完整口径”输出

```powershell
python .\causal_analysis.py
```

运行完成后，查看 `output/` 目录。

## 主要输出文件（output/目录）

- `descriptive_stats_full.csv`：完整口径描述性统计（用于论文表4-1）
- `ols_results.csv`：OLS 回归结果
- `psm_results.csv`：PSM 结果
- `pca_loadings.csv`：AI_Index PCA 载荷
- `causal_forest_proxy_summary.csv`：因果森林替代（T-learner RF）汇总
- `heterogeneity_by_edu.csv`：按教育分组异质性
- `heterogeneity_by_asset.csv`：按资产分组异质性
- `method_comparison.png`：方法对比图
- `model_sample_report.csv`：样本交集与协变量选择报告
- `processed_data.csv`：基础样本（lnTFP、Y/K/M/L 等）
- `processed_data_full.csv`：完整口径样本（含 AI_Index 与户主控制变量）

## 合规说明

- 严禁上传或公开传播 CFPS 原始数据（.dta）及任何记录级输出（如 `processed_data*.csv`）。
- 仅允许在本地分析与论文撰写中使用，公开仓库仅可包含代码、流程文档与汇总级结果。

## 复现备注

- GRF/EconML 在 Windows + Python3.14 上可能需要 C++ 编译器，安装会失败。本项目默认使用无需编译的替代方法（T-learner RandomForest）来估计异质性处理效应。
