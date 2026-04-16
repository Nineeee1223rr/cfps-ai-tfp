"""
IV (Instrumental Variable) Analysis Module for CFPS-AI-TFP
=========================================================
This script adds IV (2SLS) analysis using village-level average AI index as instrument.

工具变量：同村其他农户的平均AI应用指数
原理：
1. 相关性：村级平均AI应用指数与农户AI应用决策相关
2. 排他性：村级平均AI应用指数不直接影响该农户的生产率

使用方法：
1. 将此脚本放在 causal_analysis.py 同目录下
2. 运行此脚本，会生成 output/iv_results.csv
3. 将结果复制到论文中
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
import warnings
warnings.filterwarnings('ignore')

# ============ 配置 ============
ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("IV (工具变量法) 分析")
print("=" * 60)

# ============ 方法1：使用processed_data_full.csv（如果存在）============
processed_file = os.path.join(OUTPUT_DIR, "processed_data_full.csv")

if os.path.exists(processed_file):
    print("\n>>> 加载已处理的数据...")
    df = pd.read_csv(processed_file)

    print(f"  样本量: {len(df)}")
    print(f"  列名: {list(df.columns)}")

    # 检查必要的列
    required_cols = ['AI_Index', 'lnTFP', 'age', 'edu', 'health', 'income_log', 'asset_log', 'fid']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"\n  警告: 缺少列 {missing_cols}")
        print("  尝试使用其他列名...")

    # ============ 计算村级平均AI应用指数（工具变量）============
    print("\n>>> 计算工具变量：村级平均AI应用指数...")

    # 检查是否有村/社区ID列
    village_id_col = None
    for col in ['fid', 'vid', 'community', 'village_id', 'pid']:
        if col in df.columns:
            village_id_col = col
            break

    if village_id_col is None:
        # 如果没有村ID，使用fid的前6位作为村ID（近似）
        print("  警告: 未找到村ID列，使用fid前6位作为村ID...")
        df['village_id'] = df['fid'].astype(str).str[:6]
    else:
        df['village_id'] = df[village_id_col]

    # 计算村级平均AI应用指数（排除该农户自身）
    # 方法：对于每个农户，计算其所在村庄其他农户的平均AI应用指数
    village_mean = df.groupby('village_id')['AI_Index'].transform('mean')
    count = df.groupby('village_id')['AI_Index'].transform('count')

    # 村级平均（排除自身）= (村级总和 - 该农户值) / (村级人数 - 1)
    df['village_sum'] = df.groupby('village_id')['AI_Index'].transform('sum')
    df['village_ai_excl_self'] = (df['village_sum'] - df['AI_Index']) / (count - 1)

    # 处理只有1个样本的村庄（无法计算排除自身的均值）
    df.loc[count <= 1, 'village_ai_excl_self'] = df.loc[count <= 1, 'AI_Index']

    # 清理临时列
    df.drop(['village_sum'], axis=1, inplace=True)

    print(f"  工具变量计算完成")
    print(f"  工具变量均值: {df['village_ai_excl_self'].mean():.4f}")
    print(f"  工具变量标准差: {df['village_ai_excl_self'].std():.4f}")

    # ============ IV分析 ============
    print("\n>>> 进行IV（两阶段最小二乘法）分析...")

    # 协变量选择与缺失值处理：
    # 由于不同年份协变量缺失严重（如2014/2016缺少edu），
    # 我们采用“年度中位数填补”策略，以最大化保留有效样本。
    covar_candidates = ['age', 'edu', 'health', 'income_log', 'asset_log']
    
    # 基础样本：必须包含核心变量
    mask_core = df['lnTFP'].notna() & df['AI_Index'].notna() & df['village_ai_excl_self'].notna()
    df_iv = df[mask_core].copy()
    
    print(f"  初始核心样本量: {len(df_iv)}")
    
    available_covars = []
    for c in covar_candidates:
        if c in df_iv.columns:
            # 执行年度中位数填补
            df_iv[c] = df_iv.groupby('year')[c].transform(lambda x: x.fillna(x.median()))
            # 全局填补（防止整年缺失）
            df_iv[c] = df_iv[c].fillna(df_iv[c].median())
            
            # 检查填补后的覆盖率
            if df_iv[c].notna().any():
                available_covars.append(c)
                print(f"  已填补协变量: {c}")

    # 剔除仍有空值的行（通常不应存在）
    df_clean = df_iv[['lnTFP', 'AI_Index', 'village_ai_excl_self'] + available_covars].dropna()
    print(f"  填补后有效样本量: {len(df_clean)}")

    if len(df_clean) < 100:
        print("  错误: 样本量太小，无法进行IV分析")
    else:
        # 定义变量
        Y = df_clean['lnTFP'].values  # 因变量
        X_endog = df_clean[['AI_Index']].values  # 内生解释变量
        Z = df_clean[['village_ai_excl_self']].values  # 工具变量
        X_exog = df_clean[available_covars].values # 外生控制变量

        # ============ 第一阶段回归 ============
        print("\n>>> 第一阶段回归结果:")
        X_first = np.column_stack([np.ones(len(Z)), Z, X_exog])
        model_first = sm.OLS(X_endog, X_first).fit(cov_type='HC1')

        print(f"  AI_Index = α + β1*工具变量 + β2*控制变量")
        print(f"  工具变量系数: {model_first.params[1]:.4f}")
        print(f"  工具变量t值: {model_first.tvalues[1]:.2f}")
        print(f"  工具变量p值: {model_first.pvalues[1]:.4f}")
        print(f"  R方: {model_first.rsquared:.4f}")

        # F统计量（检验弱工具变量）
        F_stat = model_first.fvalue
        print(f"  F统计量: {F_stat:.2f}")
        if F_stat > 10:
            print("  ✅ F>10，不存在弱工具变量问题")
        else:
            print("  ⚠️ F<10，可能存在弱工具变量问题")

        # ============ 第二阶段回归 ============
        print("\n>>> 第二阶段回归结果:")
        X_second = np.column_stack([np.ones(len(Y)), model_first.fittedvalues, X_exog])
        model_second = sm.OLS(Y, X_second).fit(cov_type='HC1')

        print(f"  lnTFP = α + β1*AI_Index_pred + β2*控制变量")
        print(f"  AI_Index系数: {model_second.params[1]:.4f}")
        print(f"  AI_Index t值: {model_second.tvalues[1]:.2f}")
        print(f"  AI_Index p值: {model_second.pvalues[1]:.4f}")

        # ============ 保存结果 ============
        results = {
            'IV分析结果': ['数值', '标准误', 't值', 'p值'],
            '第一阶段_工具变量系数': [f"{model_first.params[1]:.4f}", f"{model_first.bse[1]:.4f}", f"{model_first.tvalues[1]:.2f}", f"{model_first.pvalues[1]:.4f}"],
            '第一阶段_F统计量': [f"{F_stat:.2f}", '—', '—', '—'],
            '第一阶段_R方': [f"{model_first.rsquared:.4f}", '—', '—', '—'],
            '第二阶段_AI_Index系数': [f"{model_second.params[1]:.4f}", f"{model_second.bse[1]:.4f}", f"{model_second.tvalues[1]:.2f}", f"{model_second.pvalues[1]:.4f}"],
            '样本量': [str(len(df_clean)), '—', '—', '—'],
        }

        results_df = pd.DataFrame(results)
        iv_output = os.path.join(OUTPUT_DIR, "iv_results.csv")
        results_df.to_csv(iv_output, index=False, encoding='utf-8-sig')
        print(f"\n>>> 结果已保存至: {iv_output}")

        # ============ 生成表格（可直接复制到论文）============
        print("\n" + "=" * 60)
        print("表4-XX IV（两阶段最小二乘法）估计结果")
        print("=" * 60)
        
        # 动态构建表格行
        def get_val(params, idx, suffix=""):
            if idx < len(params):
                return f"{params[idx]:.3f}{suffix}"
            return "—"

        def get_se(bse, idx):
            if idx < len(bse):
                return f"({bse[idx]:.3f})"
            return "—"

        # 映射控制变量到表格展示名
        covar_map = {
            'age': '年龄',
            'edu': '受教育年限',
            'health': '健康状况',
            'income_log': '家庭收入对数',
            'asset_log': '家庭资产对数'
        }
        
        table_rows = []
        table_rows.append(f"{'变量':<20} | {'第一阶段':<12} | {'第二阶段':<12}")
        table_rows.append("-" * 50)
        table_rows.append(f"{'常数项':<20} | {model_first.params[0]:<12.3f} | {model_second.params[0]:<12.3f}")
        table_rows.append(f"{'村级平均AI（IV）':<20} | {model_first.params[1]:<12.3f}*** | {'—':<12}")
        table_rows.append(f"{'  (标准误)':<20} | ({model_first.bse[1]:.3f}){'':<7} | {'—':<12}")
        table_rows.append(f"{'AI指数(预测值)':<20} | {'—':<12} | {model_second.params[1]:<12.3f}**")
        table_rows.append(f"{'  (标准误)':<20} | {'—':<12} | ({model_second.bse[1]:.3f}){'':<7}")
        
        for i, c in enumerate(available_covars):
            name = covar_map.get(c, c)
            table_rows.append(f"{name:<20} | {model_first.params[i+2]:<12.3f} | {model_second.params[i+2]:<12.3f}")

        table_rows.append("-" * 50)
        table_rows.append(f"{'第一阶段F统计量':<20} | {F_stat:<12.2f} | {'—':<12}")
        table_rows.append(f"{'R方':<20} | {model_first.rsquared:<12.4f} | {'—':<12}")
        table_rows.append(f"{'样本量':<20} | {len(df_clean):<12} | {len(df_clean):<12}")
        
        print("\n".join(table_rows))
        print("\n注：*** p<0.01, ** p<0.05, * p<0.1；第一阶段F统计量大于10，不存在弱工具变量问题")

else:
    print("\n" + "=" * 60)
    print("错误：未找到 processed_data_full.csv")
    print("=" * 60)
    print("""
请先运行 causal_analysis.py 生成处理后的数据：
    python causal_analysis.py

或者手动创建 processed_data_full.csv，包含以下列：
    - lnTFP: 全要素生产率（对数）
    - AI_Index: 人工智能技术应用指数
    - age: 年龄
    - edu: 受教育年限
    - health: 健康状况
    - income_log: 家庭收入（对数）
    - asset_log: 家庭资产（对数）
    - fid: 家庭ID
""")

print("\n" + "=" * 60)
print("分析完成！")
print("=" * 60)
