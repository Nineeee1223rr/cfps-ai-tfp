import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyreadstat

import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "output")


MISSING_CODES = {-10, -9, -8, -2, -1}


@dataclass
class YearSpec:
    year: int
    file: str
    # candidate household id columns, take the first found
    fid_candidates: List[str]


YEAR_SPECS = [
    YearSpec(2010, "cfps2010famecon_202008.dta", ["fid", "fid10", "fid_base"]),
    YearSpec(2012, "cfps2012famecon_201906.dta", ["fid", "fid12", "fid10", "fid_base"]),
    YearSpec(2014, "cfps2014famecon_201906.dta", ["fid", "fid14", "fid10", "fid_base"]),
    YearSpec(2016, "cfps2016famecon_201807.dta", ["fid", "fid16", "fid10", "fid_base"]),
    YearSpec(2018, "cfps2018famecon_202101.dta", ["fid", "fid18", "fid10", "fid_base"]),
    YearSpec(2020, "cfps2020famecon_202306.dta", ["fid", "fid20", "fid10", "fid_base"]),
]


def _read_meta(path: str):
    _df, meta = pyreadstat.read_dta(path, metadataonly=True)
    return meta


def _safe_mean(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce")
    return float(s.mean(skipna=True)) if s.notna().any() else float("nan")


def _first_existing(col_candidates: List[str], columns: List[str]) -> Optional[str]:
    cols = set(columns)
    for c in col_candidates:
        if c in cols:
            return c
    return None


def _coerce_numeric(s: pd.Series) -> pd.Series:
    # coerce to numeric, keep NaN for non-numeric
    return pd.to_numeric(s, errors="coerce")


def _replace_missing_codes(s: pd.Series) -> pd.Series:
    s = _coerce_numeric(s)
    return s.mask(s.isin(MISSING_CODES))


def _sum_cols(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series(np.nan, index=df.index)
    out = pd.Series(0.0, index=df.index)
    has_any = pd.Series(False, index=df.index)
    for c in existing:
        v = _replace_missing_codes(df[c])
        has_any = has_any | v.notna()
        out = out.add(v.fillna(0.0))
    out = out.mask(~has_any)
    return out


def _positive_log(x: pd.Series) -> pd.Series:
    x = _coerce_numeric(x)
    return np.log(x.where(x > 0))


def _ols_residual(y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # y: (n,), X: (n, k)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, resid


def build_year_df(spec: YearSpec) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, spec.file)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    meta = _read_meta(path)
    cols = list(meta.column_names)

    fid_col = _first_existing(spec.fid_candidates, cols)
    if fid_col is None:
        raise RuntimeError(f"No fid column found in {spec.file}. Tried {spec.fid_candidates}")

    # Variables for agricultural output and inputs differ by wave.
    # We load a superset and compute by availability.
    usecols = {
        fid_col,
        # output candidates
        "inc_agri",
        "fk3",
        "fl3",
        "fl6",
        "fl7",
        "fl8",
        "fl9",
        "fl10",
        # inputs
        "fixed_asset",
        # crop inputs
        "fl401",
        "fl402",
        "fl403",
        "fl404",
        "fl501",
        "fl502",
        "fl503",
        "fl504",
        "fl505",
        # livestock inputs
        "fl901",
        "fl902",
        "fl903",
        "fl904",
        "fl801",
        "fl802",
        "fl803",
        "fl804",
        "fl805",
        # household income/asset controls (where available)
        "finc",
        "finc1",
        "total_asset",
        "asset",
        "finance_asset",
        "land_asset",
        # head pid if exists (future use)
        "headpid",
        "resp1pid",
        "fresp1pid",
        "fl2pid",
        "resp2pid",
        "fresp2pid",
        "fz1pid",
    }
    usecols_existing = [c for c in usecols if c in cols]

    df, _meta2 = pyreadstat.read_dta(path, usecols=usecols_existing)
    df = df.rename(columns={fid_col: "fid"})
    df["year"] = spec.year

    # Head pid mapping (priority order)
    head_pid = None
    for c in ["headpid", "resp1pid", "fresp1pid", "fl2pid", "resp2pid", "fresp2pid", "fz1pid"]:
        if c in df.columns:
            head_pid = df[c]
            break
    if head_pid is None:
        head_pid = pd.Series(np.nan, index=df.index)

    # output Y
    if "inc_agri" in df.columns:
        y = _replace_missing_codes(df["inc_agri"])
    elif "fk3" in df.columns:
        y = _replace_missing_codes(df["fk3"])
    elif "fl9" in df.columns:
        y = _replace_missing_codes(df["fl9"])
    elif "fl3" in df.columns:
        # 2012 wave structure: crop/forestry total value + livestock/aquatic value + byproducts
        y = _sum_cols(df, ["fl3", "fl7", "fl8"])
    else:
        y = pd.Series(np.nan, index=df.index)

    # optional add self-consumption value
    y_self = _replace_missing_codes(df["fl10"]) if "fl10" in df.columns else pd.Series(np.nan, index=df.index)

    # capital K
    K = _replace_missing_codes(df["fixed_asset"]) if "fixed_asset" in df.columns else pd.Series(np.nan, index=df.index)

    # intermediate inputs M (seed/fertilizer/pesticide + irrigation + machine rental + other + livestock feed etc)
    m_crop = _sum_cols(df, ["fl401", "fl403", "fl501", "fl503", "fl504", "fl505", "fl404"])  # wave-specific
    m_liv = _sum_cols(df, ["fl901", "fl903", "fl904", "fl801", "fl803", "fl804", "fl805"])  # wave-specific
    M = _sum_cols(pd.DataFrame({"m_crop": m_crop, "m_liv": m_liv}), ["m_crop", "m_liv"])

    # labor proxy L: use hired labor cost where available
    L_cost = _sum_cols(df, ["fl402", "fl502", "fl802", "fl902"])  # wave-specific

    # household income & assets (for later controls)
    income = None
    for c in ["finc1", "finc"]:
        if c in df.columns:
            income = _replace_missing_codes(df[c])
            break
    if income is None:
        income = pd.Series(np.nan, index=df.index)

    asset = None
    for c in ["total_asset", "asset", "finance_asset", "land_asset"]:
        if c in df.columns:
            asset = _replace_missing_codes(df[c])
            break
    if asset is None:
        asset = pd.Series(np.nan, index=df.index)

    out = pd.DataFrame(
        {
            "fid": df["fid"],
            "year": df["year"],
            "head_pid": head_pid,
            "Y": y,
            "Y_self": y_self,
            "K": K,
            "M": M,
            "L_cost": L_cost,
            "income": income,
            "asset": asset,
        }
    )

    # de-duplicate within year
    out = out.drop_duplicates(subset=["fid", "year"])
    return out


def compute_tfp(df: pd.DataFrame) -> pd.DataFrame:
    # Use OLS production function residual as lnTFP.
    # lnY = b0 + bK lnK + bM lnM + bL lnL_cost + e
    df = df.copy()

    df["lnY"] = _positive_log(df["Y"])
    df["lnK"] = _positive_log(df["K"])
    df["lnM"] = _positive_log(df["M"])
    df["lnL"] = _positive_log(df["L_cost"].fillna(0) + 1.0)

    reg = df[["lnY", "lnK", "lnM", "lnL"]].dropna()
    if len(reg) < 200:
        df["lnTFP"] = np.nan
        df["tfp_note"] = "insufficient_obs_for_ols"
        return df

    y = reg["lnY"].to_numpy()
    X = np.column_stack(
        [
            np.ones(len(reg)),
            reg["lnK"].to_numpy(),
            reg["lnM"].to_numpy(),
            reg["lnL"].to_numpy(),
        ]
    )

    beta, resid = _ols_residual(y, X)
    df.loc[reg.index, "lnTFP"] = resid
    df["tfp_b0"] = beta[0]
    df["tfp_bK"] = beta[1]
    df["tfp_bM"] = beta[2]
    df["tfp_bL"] = beta[3]
    df["tfp_note"] = "ols_residual"
    return df


def _read_person_controls(year: int) -> pd.DataFrame:
    """Read person/adult file for a given wave and return pid-level controls and digital vars."""
    if year == 2010:
        file = "cfps2010adult_202008.dta"
        pid_col = "pid"
        cols = [
            "pid",
            "qa1age",
            "cfps2010eduy_best",
            "qp201",
            "qz202",
            "ku1",
            "ku2",
            "ku301",
            "ku302",
            "ku303",
            "ku304",
            "kt3_a_1",
            "kt3_a_2",
        ]
        age_col = "qa1age"
        edu_col = "cfps2010eduy_best"
        health_col = "qp201" if True else "qz202"
        train_cols = ["kt3_a_1", "kt3_a_2"]
        digital_cols = ["ku1", "ku2", "ku301", "ku302", "ku303", "ku304"]
    elif year == 2012:
        file = "cfps2012adult_201906.dta"
        pid_col = "pid"
        cols = [
            "pid",
            "qv201b",
            "qc201",
            "qp201",
            "qz202",
            "qg104y",
            "qg104m",
        ]
        age_col = "qv201b"
        edu_col = "qc201"
        health_col = "qp201"
        train_cols = ["qg104y", "qg104m"]
        digital_cols = []
    elif year == 2014:
        file = "cfps2014adult_201906.dta"
        pid_col = "pid"
        cols = [
            "pid",
            "cfps2014_age",
            "cfps2014eduy_best",
            "qp201",
            "qz202",
            "qgb2",
            "ku1m",
            "ku701",
            "ku702",
            "ku703",
            "ku704",
            "ku705",
            "ku802",
        ]
        age_col = "cfps2014_age"
        edu_col = "cfps2014eduy_best" if True else "qc201"
        health_col = "qp201"
        train_cols = ["qgb2"]
        digital_cols = ["ku1m", "ku701", "ku702", "ku703", "ku704", "ku705", "ku802"]
    elif year == 2016:
        file = "cfps2016adult_201906.dta"
        pid_col = "pid"
        cols = [
            "pid",
            "cfps2016_age",
            "cfps2016eduy_best",
            "qp201",
            "qz202",
            "qgb2",
            "ku1m",
            "ku701",
            "ku702",
            "ku703",
            "ku704",
            "ku705",
            "ku802",
        ]
        age_col = "cfps2016_age"
        edu_col = "cfps2016eduy_best"
        health_col = "qp201"
        train_cols = ["qgb2"]
        digital_cols = ["ku1m", "ku701", "ku702", "ku703", "ku704", "ku705", "ku802"]
    elif year == 2018:
        file = "cfps2018person_202012.dta"
        pid_col = "pid"
        cols = [
            "pid",
            "age",
            "cfps2018eduy_im",
            "qp201",
            "qz202",
            "qgb2",
            "qu1m",
            "qu201",
            "qu202",
            "qu701",
            "qu702",
            "qu703",
            "qu704",
            "qu705",
            "qu802",
            "qn202",
        ]
        age_col = "age"
        edu_col = "cfps2018eduy_im"
        health_col = "qp201"
        train_cols = ["qgb2"]
        digital_cols = [
            "qu1m",
            "qu201",
            "qu202",
            "qu701",
            "qu702",
            "qu703",
            "qu704",
            "qu705",
            "qu802",
            "qn202",
        ]
    elif year == 2020:
        file = "cfps2020person_202112.dta"
        pid_col = "pid"
        cols = [
            "pid",
            "age",
            "cfps2020eduy_im",
            "qz202",
            "qgb2",
            "qu201",
            "qu202",
            "qu201a",
            "qu202a",
            "qu91",
            "qu94",
            "qu802",
            "qn202",
        ]
        age_col = "age"
        edu_col = "cfps2020eduy_im"
        health_col = "qz202"
        train_cols = ["qgb2"]
        digital_cols = [
            "qu201",
            "qu202",
            "qu201a",
            "qu202a",
            "qu91",
            "qu94",
            "qu802",
            "qn202",
        ]
    else:
        raise ValueError(year)

    path = os.path.join(DATA_DIR, file)
    meta = _read_meta(path)
    existing = [c for c in cols if c in meta.column_names]
    df, _m2 = pyreadstat.read_dta(path, usecols=existing)

    out = pd.DataFrame({"pid": df[pid_col]})
    if age_col in df.columns:
        out["age"] = _replace_missing_codes(df[age_col])
    else:
        out["age"] = np.nan
    if edu_col in df.columns:
        out["edu"] = _replace_missing_codes(df[edu_col])
    else:
        out["edu"] = np.nan
    if health_col in df.columns:
        out["health"] = _replace_missing_codes(df[health_col])
    else:
        out["health"] = np.nan

    # training: any non-missing positive signal
    train = pd.Series(0.0, index=df.index)
    has_train = pd.Series(False, index=df.index)
    for c in train_cols:
        if c in df.columns:
            v = _replace_missing_codes(df[c])
            has_train = has_train | v.notna()
            # binary questions are typically 1=yes
            train = np.where(v.fillna(0) > 0, 1.0, train)
    out["train"] = np.where(has_train, train, np.nan)

    # digital vars
    for c in digital_cols:
        if c in df.columns:
            out[c] = _replace_missing_codes(df[c])

    out = out.dropna(subset=["pid"]).drop_duplicates(subset=["pid"])
    return out


def _compute_ai_index(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute AI_Index per year using PCA on available digital variables; fallback to standardized sum."""
    panel = panel.copy()
    loadings_rows = []

    panel["AI_Index"] = np.nan
    panel["ai_method"] = "missing"

    # candidate features by year (those present in panel)
    for year, g in panel.groupby("year"):
        candidates = [c for c in g.columns if c.startswith("ku") or c.startswith("qu") or c == "qn202"]
        # exclude obvious non-digital
        candidates = [c for c in candidates if c not in {"ku102", "ku102a", "qu201a", "qu202a"}]
        # keep only numeric
        X = g[candidates].apply(pd.to_numeric, errors="coerce") if candidates else pd.DataFrame(index=g.index)
        # drop columns with too many missing
        keep = []
        for c in X.columns:
            if X[c].notna().sum() >= max(200, int(0.1 * len(g))):
                keep.append(c)
        X = X[keep]

        if X.shape[1] >= 2:
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X.fillna(X.median(numeric_only=True)))
            pca = PCA(n_components=1, random_state=42)
            pc1 = pca.fit_transform(Xs).reshape(-1)
            # standardize to mean 0 std 1 within year
            pc1 = (pc1 - np.mean(pc1)) / (np.std(pc1) + 1e-9)
            panel.loc[g.index, "AI_Index"] = pc1
            panel.loc[g.index, "ai_method"] = "pca_year"
            for var, loading in zip(X.columns, pca.components_[0]):
                loadings_rows.append(
                    {
                        "year": int(year),
                        "var": var,
                        "loading": float(loading),
                        "explained_var": float(pca.explained_variance_ratio_[0]),
                    }
                )
        elif X.shape[1] == 1:
            v = X.iloc[:, 0]
            z = (v - v.mean(skipna=True)) / (v.std(skipna=True) + 1e-9)
            panel.loc[g.index, "AI_Index"] = z
            panel.loc[g.index, "ai_method"] = "z_single"
            loadings_rows.append({"year": int(year), "var": X.columns[0], "loading": 1.0, "explained_var": 1.0})
        else:
            # no vars
            continue

    loadings = pd.DataFrame(loadings_rows)
    return panel, loadings


def _ols(panel: pd.DataFrame, y: str, x: List[str]) -> pd.DataFrame:
    d = panel[[y] + x].dropna().copy()
    if len(d) < 500:
        return pd.DataFrame([{ "note": "insufficient_obs" }])
    Y = d[y]
    X = sm.add_constant(d[x], has_constant="add")
    model = sm.OLS(Y, X).fit(cov_type="HC1")
    rows = []
    for name in model.params.index:
        rows.append(
            {
                "term": name,
                "coef": float(model.params[name]),
                "se": float(model.bse[name]),
                "t": float(model.tvalues[name]),
                "p": float(model.pvalues[name]),
                "n": int(model.nobs),
                "r2": float(model.rsquared),
            }
        )
    return pd.DataFrame(rows)


def _choose_covariates(panel: pd.DataFrame, outcome: str, main: str, candidates: List[str], min_n: int = 2000) -> Tuple[List[str], pd.DataFrame]:
    """Greedy select covariates to keep overlap >= min_n."""
    report_rows = []
    base = panel[outcome].notna() & panel[main].notna()
    base_n = int(base.sum())
    report_rows.append({"step": "base", "var": "(outcome & main)", "n": base_n})

    # score candidates by how much overlap they keep
    scores = []
    for c in candidates:
        if c not in panel.columns:
            continue
        n = int((base & panel[c].notna()).sum())
        scores.append((n, c))
        report_rows.append({"step": "single", "var": c, "n": n})

    # prefer covariates with higher overlap
    scores.sort(reverse=True)
    chosen: List[str] = []
    mask = base.copy()
    for n_single, c in scores:
        new_mask = mask & panel[c].notna()
        new_n = int(new_mask.sum())
        # only add if still enough observations
        if new_n >= min_n:
            chosen.append(c)
            mask = new_mask
            report_rows.append({"step": "add", "var": c, "n": new_n})

    report_rows.append({"step": "final", "var": "+".join(chosen) if chosen else "(none)", "n": int(mask.sum())})
    return chosen, pd.DataFrame(report_rows)


def _psm_ate(panel: pd.DataFrame, outcome: str, treat: str, covars: List[str]) -> Dict[str, float]:
    d = panel[[outcome, treat] + covars].dropna().copy()
    if len(d) < 1000:
        return {"note": "insufficient_obs"}

    y = d[outcome].to_numpy()
    t = d[treat].to_numpy().astype(int)
    X = d[covars].to_numpy()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    lr = LogisticRegression(max_iter=2000)
    lr.fit(Xs, t)
    ps = lr.predict_proba(Xs)[:, 1]

    treated_idx = np.where(t == 1)[0]
    control_idx = np.where(t == 0)[0]
    if len(treated_idx) < 200 or len(control_idx) < 200:
        return {"note": "insufficient_treated_or_control"}

    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(ps[control_idx].reshape(-1, 1))
    dist, ind = nn.kneighbors(ps[treated_idx].reshape(-1, 1))
    matched_controls = control_idx[ind.reshape(-1)]

    att = float(np.mean(y[treated_idx] - y[matched_controls]))
    return {
        "att": att,
        "n": float(len(d)),
        "treated": float(len(treated_idx)),
        "control": float(len(control_idx)),
        "ps_mean": float(np.mean(ps)),
    }


def _t_learner_rf(panel: pd.DataFrame, outcome: str, treat: str, covars: List[str]) -> pd.Series:
    d = panel[[outcome, treat] + covars].dropna().copy()
    if len(d) < 2000:
        return pd.Series(np.nan, index=panel.index)

    y = d[outcome]
    t = d[treat].astype(int)
    X = d[covars]

    rf_t = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1, min_samples_leaf=50)
    rf_c = RandomForestRegressor(n_estimators=200, random_state=43, n_jobs=-1, min_samples_leaf=50)

    rf_t.fit(X[t == 1], y[t == 1])
    rf_c.fit(X[t == 0], y[t == 0])

    mu1 = rf_t.predict(X)
    mu0 = rf_c.predict(X)
    tau = mu1 - mu0

    out = pd.Series(np.nan, index=panel.index)
    out.loc[d.index] = tau
    return out


def descriptive_stats(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        rows.append(
            {
                "variable": c,
                "n": int(s.notna().sum()),
                "mean": float(s.mean(skipna=True)) if s.notna().any() else np.nan,
                "std": float(s.std(skipna=True)) if s.notna().any() else np.nan,
                "min": float(s.min(skipna=True)) if s.notna().any() else np.nan,
                "max": float(s.max(skipna=True)) if s.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    frames = []
    for spec in YEAR_SPECS:
        frames.append(build_year_df(spec))

    df = pd.concat(frames, ignore_index=True)

    # Keep likely agricultural households: positive output OR positive crop input
    df = df[(df["Y"].fillna(0) > 0) | (df["M"].fillna(0) > 0) | (df["K"].fillna(0) > 0)].copy()

    df = compute_tfp(df)

    # Merge head controls per year
    controls = []
    for y in sorted(df["year"].unique()):
        try:
            c = _read_person_controls(int(y))
            c["year"] = int(y)
            controls.append(c)
        except Exception:
            continue
    if controls:
        cdf = pd.concat(controls, ignore_index=True)
    else:
        cdf = pd.DataFrame(columns=["pid", "year", "age", "edu", "health", "train"])

    df = df.merge(cdf, how="left", left_on=["head_pid", "year"], right_on=["pid", "year"], suffixes=("", "_p"))
    df = df.drop(columns=[c for c in ["pid"] if c in df.columns])

    # AI_Index from digital columns
    df, loadings = _compute_ai_index(df)

    # Add land/transfer proxies from famecon land modules if present in processed
    # We approximate land with land_asset when available.
    df["land"] = df["asset"].where(df["asset"].notna())
    df["income_log"] = _positive_log(df["income"])
    df["asset_log"] = _positive_log(df["asset"].where(df["asset"] > 0))

    processed_full = os.path.join(OUTPUT_DIR, "processed_data_full.csv")
    df.to_csv(processed_full, index=False, encoding="utf-8-sig")

    loadings_path = os.path.join(OUTPUT_DIR, "pca_loadings.csv")
    loadings.to_csv(loadings_path, index=False, encoding="utf-8-sig")

    # Descriptive stats for key thesis variables
    stats_cols = ["lnTFP", "AI_Index", "age", "edu", "health", "train", "Y", "K", "M", "L_cost", "income", "asset"]
    stats = descriptive_stats(df, stats_cols)
    stats_path = os.path.join(OUTPUT_DIR, "descriptive_stats_full.csv")
    stats.to_csv(stats_path, index=False, encoding="utf-8-sig")

    # OLS baseline
    # Prefer full-thesis controls but adaptively drop those that destroy overlap.
    covar_candidates = [
        "age",
        "edu",
        "health",
        "train",
        "income_log",
        "asset_log",
        "income",
        "asset",
    ]
    covar_candidates = [c for c in covar_candidates if c in df.columns]
    chosen_covars, sample_report = _choose_covariates(df, outcome="lnTFP", main="AI_Index", candidates=covar_candidates, min_n=2000)
    sample_report.to_csv(os.path.join(OUTPUT_DIR, "model_sample_report.csv"), index=False, encoding="utf-8-sig")

    ols_res = _ols(df, "lnTFP", chosen_covars + ["AI_Index"])
    ols_path = os.path.join(OUTPUT_DIR, "ols_results.csv")
    ols_res.to_csv(ols_path, index=False, encoding="utf-8-sig")

    # PSM on binary treatment
    dpsm = df.copy()
    med = dpsm["AI_Index"].median(skipna=True)
    dpsm["treat"] = (dpsm["AI_Index"] > med).astype(float)
    psm_covars = chosen_covars
    psm = _psm_ate(dpsm, "lnTFP", "treat", psm_covars)
    psm_path = os.path.join(OUTPUT_DIR, "psm_results.csv")
    pd.DataFrame([psm]).to_csv(psm_path, index=False, encoding="utf-8-sig")

    # Causal forest proxy: T-learner RF
    tau = _t_learner_rf(dpsm, "lnTFP", "treat", psm_covars)
    dpsm["tau_hat"] = tau
    tau_stats = {
        "tau_mean": _safe_mean(dpsm["tau_hat"]),
        "tau_p10": float(pd.to_numeric(dpsm["tau_hat"], errors="coerce").quantile(0.1)),
        "tau_p50": float(pd.to_numeric(dpsm["tau_hat"], errors="coerce").quantile(0.5)),
        "tau_p90": float(pd.to_numeric(dpsm["tau_hat"], errors="coerce").quantile(0.9)),
    }
    tau_path = os.path.join(OUTPUT_DIR, "causal_forest_proxy_summary.csv")
    pd.DataFrame([tau_stats]).to_csv(tau_path, index=False, encoding="utf-8-sig")

    # simple comparison plot
    fig_path = os.path.join(OUTPUT_DIR, "method_comparison.png")
    methods = ["OLS(beta_AI)", "PSM(ATT)", "CF_proxy(tau_mean)"]
    ols_ai = float(ols_res.loc[ols_res["term"] == "AI_Index", "coef"].iloc[0]) if ("term" in ols_res.columns and (ols_res["term"] == "AI_Index").any()) else np.nan
    vals = [ols_ai, float(psm.get("att", np.nan)), float(tau_stats.get("tau_mean", np.nan))]
    plt.figure(figsize=(7, 4))
    plt.bar(methods, vals)
    plt.ylabel("Effect on lnTFP")
    plt.title("Method Comparison")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    # heterogeneity tables (education quartiles)
    het = dpsm[["tau_hat", "edu", "asset_log"]].copy()
    het["edu_group"] = pd.qcut(pd.to_numeric(het["edu"], errors="coerce"), 4, duplicates="drop")
    het["asset_group"] = pd.qcut(pd.to_numeric(het["asset_log"], errors="coerce"), 4, duplicates="drop")
    het_edu = het.groupby("edu_group")["tau_hat"].agg(["count", "mean", "std"]).reset_index()
    het_asset = het.groupby("asset_group")["tau_hat"].agg(["count", "mean", "std"]).reset_index()
    het_edu.to_csv(os.path.join(OUTPUT_DIR, "heterogeneity_by_edu.csv"), index=False, encoding="utf-8-sig")
    het_asset.to_csv(os.path.join(OUTPUT_DIR, "heterogeneity_by_asset.csv"), index=False, encoding="utf-8-sig")

    print(f"Wrote: {processed_full}")
    print(f"Wrote: {stats_path}")
    print(f"Wrote: {loadings_path}")
    print(f"Wrote: {ols_path}")
    print(f"Wrote: {psm_path}")
    print(f"Wrote: {tau_path}")
    print(f"Wrote: {fig_path}")


if __name__ == "__main__":
    main()
