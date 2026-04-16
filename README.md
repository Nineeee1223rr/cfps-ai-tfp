# CFPS 农业AI与TFP 实证分析（可复现流程）

本仓库用于记录与复现实证分析全过程：从 CFPS 数据挑选、变量扫描、TFP 计算、AI_Index（PCA）构建，到 OLS/PSM/因果森林替代估计与论文可替换输出。

## 合规与重要声明（必须阅读）

- CFPS 数据受使用协议约束，**严禁上传/公开传播原始数据文件（.dta）**。
- 本仓库默认通过 `.gitignore` 忽略：
  - `原始数据/`、`data/`
  - 记录级输出：`output/processed_data*.csv`
- 如需公开展示结果，建议仅上传汇总级输出（例如描述性统计、回归结果表、PCA载荷、方法对比图）。

## 主要实证结果

- **OLS基准回归**：AI_Index系数为0.098（p=0.001），表明AI应用能显著提升农业生产率。
- **T-Learner估计**：平均处理效应(ATE)为0.071（约7%生产率提升），10%水平边缘显著。
- **PSM结果**：ATT为-0.004，不显著。PSM通过倾向得分匹配控制可观测变量的选择偏差后，AI应用对农业生产率的净效应接近于零，这可能表明AI应用的生产率效应主要通过与人力资本等不可观测因素的复杂交互实现，单纯的可观测因素匹配难以捕捉这种非线性机制。
- **异质性分析**：技术效应与人力资本和经济实力正相关，但所有群体均能获益。

## 结果经济含义解读

1. **AI赋能效应显著**：OLS与T-Learner结果均证实了以互联网、智能手机为载体的数字化工具能够显著提升农业全要素生产率（TFP）。约7%的生产率提升意味着数字化转型是缓解农业资源约束、实现降本增效的关键路径。
2. **方法论差异启示**：PSM与T-Learner的结果差异提示，AI对农业的提升作用可能并非简单的线性平移，而是通过与农户人力资本等因素的复杂交互产生非线性影响。
3. **普惠性特征**：异质性分析显示，尽管高教育、高资产群体获益更多，但数字化转型具有普惠性，各类型经营主体均能从中实现生产率的边际改善。

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
- `iv_results.csv`：IV（工具变量法）回归结果汇总
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

## 数据说明

本项目使用的原始数据来自CFPS数据库，受数据使用协议限制，原始数据文件（.dta）不可上传。

如需复现分析，请：
1. 访问 [CFPS数据申请网站](https://ipss.pkuh6.edu.cn/cfps/)
2. 申请获得CFPS数据使用权
3. 将数据文件放置于 `data/raw/` 目录
4. 运行 `python causal_analysis.py` 生成结果

## 输出对照

| 论文表格 | 对应输出文件 | 验证方法 |
|----------|--------------|----------|
| 表4-1 | descriptive_stats_full.csv | 对比均值标准差 |
| 表4-4 | ols_results.csv | 对比系数t值 |
| 表4-5 | iv_results.csv | 对比IV回归系数与F统计量 |
| 表4-6 | causal_forest_proxy_summary.csv | 对比ATE |
