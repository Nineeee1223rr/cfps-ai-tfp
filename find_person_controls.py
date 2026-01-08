import os
import re
import csv
import pyreadstat

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output", "person_control_candidates.csv")

FILES = [
    "cfps2010adult_202008.dta",
    "cfps2012adult_201906.dta",
    "cfps2014adult_201906.dta",
    "cfps2016adult_201906.dta",
    "cfps2018person_202012.dta",
    "cfps2020person_202112.dta",
]

# Keywords for paper controls / AI index proxies
KEYWORDS = [
    "年龄", "教育", "受教育", "受教育年限", "受教育程度",
    "健康", "自评健康",
    "培训", "技术培训", "农业技术培训",
    "手机", "智能手机", "互联网", "网络", "上网", "APP", "社交", "咨询", "专家", "政策", "市场", "行情", "价格", "预测",
]

# composite matcher: digital behavior
DIGITAL_WORDS = ["手机", "互联网", "网络", "上网", "APP", "社交", "咨询", "政策", "市场", "行情", "价格", "预测"]


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
                hits = [kw for kw in KEYWORDS if kw in lab]

                # add fuzzy for common variable names
                vl = lab.lower()
                vn = v.lower()
                if ("age" == vn) or ("age" in vn and "" == lab):
                    hits.append("__name_age__")
                if any(dw in lab for dw in DIGITAL_WORDS):
                    hits.append("__digital__")

                if hits:
                    w.writerow([fn, v, lab, ";".join(sorted(set(hits)))])

    print(f"Wrote: {OUT}")


if __name__ == "__main__":
    main()
