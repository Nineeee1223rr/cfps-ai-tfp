import csv
import os
import pyreadstat

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output", "y_candidates_by_year.csv")

FAMECON_FILES = [
    "cfps2010famecon_202008.dta",
    "cfps2012famecon_201906.dta",
    "cfps2014famecon_201906.dta",
    "cfps2016famecon_201807.dta",
    "cfps2018famecon_202101.dta",
    "cfps2020famecon_202306.dta",
]

KEYWORDS = [
    "农、林、牧", "农林牧", "副、渔", "渔业",
    "毛收入", "总产值", "总产出", "总收入",
    "农业生产总收入", "农业生产纯收入", "农业生产总产值",
    "农业经营", "农业收入", "农业产值",
    "种植业", "林业", "畜牧", "养殖", "渔业生产",
    "销售", "销售收入", "经营收入", "收入合计",
    "农产品", "农作物", "作物",
]

# Composite matching: label contains agri-domain AND income/output signal
AGRI_WORDS = ["农业", "农、林、牧", "农林牧", "种植", "林业", "牧", "畜", "养殖", "渔", "水产", "农产品"]
INCOME_WORDS = ["收入", "产值", "产出", "销售", "毛收入", "总收入", "总产值"]


def norm(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()


def main():
    os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)

    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "var_name", "var_label", "matched_keywords"])

        for fname in FAMECON_FILES:
            path = os.path.join(DATA_DIR, fname)
            if not os.path.exists(path):
                w.writerow([fname, "__MISSING__", "", ""])
                continue

            _df, meta = pyreadstat.read_dta(path, metadataonly=True)
            labels = getattr(meta, "column_names_to_labels", None) or {}

            for var in meta.column_names:
                lab = norm(labels.get(var, ""))
                hits = [kw for kw in KEYWORDS if kw in lab]

                # Composite: likely agricultural output/income
                is_agri = any(w in lab for w in AGRI_WORDS)
                is_income = any(w in lab for w in INCOME_WORDS)
                if is_agri and is_income and "__composite__" not in hits:
                    hits.append("__composite__")

                if hits:
                    w.writerow([fname, var, lab, ";".join(hits)])

    print(f"Wrote: {OUT}")


if __name__ == "__main__":
    main()
