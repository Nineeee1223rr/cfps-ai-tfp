import os
import csv
import pyreadstat

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output", "famconf_head_vars.csv")

FILES = [
    "cfps2010famconf_202008.dta",
    "cfps2012famconf_092015.dta",
    "cfps2014famconf_170630.dta",
    "cfps2016famconf_201804.dta",
    "cfps2018famconf_202008.dta",
    "cfps2020famconf_202301.dta",
]

KEYWORDS = [
    "户主", "家长", "与户主", "关系", "与家长", "家庭关系",
    "relation", "household head", "head",
]


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

        for fn in FILES:
            path = os.path.join(DATA_DIR, fn)
            if not os.path.exists(path):
                w.writerow([fn, "__MISSING__", "", ""])
                continue

            _df, meta = pyreadstat.read_dta(path, metadataonly=True)
            labels = getattr(meta, "column_names_to_labels", None) or {}

            for v in meta.column_names:
                lab = norm(labels.get(v, ""))
                hits = []
                low = lab.lower()
                for kw in KEYWORDS:
                    if kw.lower() in low:
                        hits.append(kw)
                if hits:
                    w.writerow([fn, v, lab, ";".join(sorted(set(hits)))])

    print(f"Wrote: {OUT}")


if __name__ == "__main__":
    main()
