import csv
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
IN_CSV = os.path.join(ROOT, "output", "var_catalog.csv")
OUT_CSV = os.path.join(ROOT, "output", "concept_candidates.csv")

# Concepts we need for the paper (lunwen.txt)
CONCEPTS = {
    "Y_ag_output": ["农业", "总产值", "产值", "产出", "种植", "林业", "牧业", "渔业"],
    "K_fixed_asset": ["固定资产", "农机", "生产性固定资产", "资产"],
    "L_labor": ["劳动", "工时", "天数", "雇工"],
    "M_intermediate": ["中间品", "种子", "化肥", "农药", "灌溉"],
    "land_area": ["耕地", "土地", "亩", "面积"],
    "plot_count": ["地块"],
    "income_total": ["总收入", "家庭总收入", "收入"],
    "asset_total": ["家庭净资产", "总资产", "资产"],
    "age": ["年龄"],
    "edu": ["受教育", "教育"],
    "health": ["健康"],
    "train": ["培训", "技术培训"],
    "coop": ["合作社"],
    "transfer": ["流转", "出租", "租用", "土地出租"],
    "internet_info": ["互联网", "网络", "上网", "政策", "市场", "行情", "价格", "预测", "APP", "社交", "咨询", "专家"],
}


def norm(s: str) -> str:
    return (s or "").strip()


def score(label: str, var: str, keywords: list[str]) -> int:
    text = (label + " " + var).lower()
    sc = 0
    for kw in keywords:
        if kw.lower() in text:
            sc += 2
    # prefer shorter, non-repeated patterns
    if re.fullmatch(r"[a-z]{1,3}\d+[a-z_]*", var.lower()):
        sc += 1
    return sc


def main():
    if not os.path.exists(IN_CSV):
        raise SystemExit(f"Missing {IN_CSV}, run scan_vars.py first")

    rows = []
    with open(IN_CSV, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            file = norm(r.get("file"))
            var_name = norm(r.get("var_name"))
            var_label = norm(r.get("var_label"))
            if not file or not var_name:
                continue
            rows.append((file, var_name, var_label))

    out = []
    # For each file+concept, keep top N
    TOPN = 25
    files = sorted({f for f, _, _ in rows})

    for f in files:
        file_rows = [(vn, vl) for ff, vn, vl in rows if ff == f]
        for concept, kws in CONCEPTS.items():
            scored = []
            for vn, vl in file_rows:
                s = score(vl, vn, kws)
                if s > 0:
                    scored.append((s, vn, vl))
            scored.sort(key=lambda x: (-x[0], x[1]))
            for rank, (s, vn, vl) in enumerate(scored[:TOPN], start=1):
                out.append({
                    "file": f,
                    "concept": concept,
                    "rank": rank,
                    "score": s,
                    "var_name": vn,
                    "var_label": vl,
                })

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "concept", "rank", "score", "var_name", "var_label"])
        writer.writeheader()
        writer.writerows(out)

    print(f"Wrote: {OUT_CSV}")


if __name__ == "__main__":
    main()
