import os
import re
import csv

import pyreadstat


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output", "agri_output_vars_by_year.csv")

FILES = [
    "cfps2010famecon_202008.dta",
    "cfps2012famecon_201906.dta",
    "cfps2014famecon_201906.dta",
    "cfps2016famecon_201807.dta",
    "cfps2018famecon_202101.dta",
    "cfps2020famecon_202306.dta",
]

# Regex: agri domain + income/output/sales signals
PAT = re.compile(
    r"(农业|农林牧|农、林、牧|种植|林业|畜牧|养殖|渔|水产|农产品).*(毛收入|收入|产值|产出|销售|总收入|总产值|总产出)"
)

# Extra: common variable names worth keeping
NAME_HINTS = re.compile(r"^(inc_agri|net_agri|fk3|fk4|fl\d+|foperate_\d+|foperate|finc\d*|fixed_asset)$", re.I)


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
        w.writerow(["file", "var_name", "var_label", "matched_by"])

        for fn in FILES:
            path = os.path.join(DATA_DIR, fn)
            if not os.path.exists(path):
                w.writerow([fn, "__MISSING__", "", ""])
                continue

            _df, meta = pyreadstat.read_dta(path, metadataonly=True)
            labels = getattr(meta, "column_names_to_labels", None) or {}

            hits = []
            for v in meta.column_names:
                lab = norm(labels.get(v, ""))
                by = []
                if lab and PAT.search(lab):
                    by.append("label_regex")
                if NAME_HINTS.search(v):
                    by.append("name_hint")
                if by:
                    hits.append((v, lab, "+".join(by)))

            hits.sort(key=lambda x: (x[0]))
            for v, lab, by in hits:
                w.writerow([fn, v, lab, by])

    print(f"Wrote: {OUT}")


if __name__ == "__main__":
    main()
