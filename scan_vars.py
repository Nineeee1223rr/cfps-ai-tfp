import os
import re
import csv
from datetime import datetime

import pyreadstat


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
OUTPUT_DIR = os.path.join(ROOT, "output")


KEYWORDS = [
    # TFP inputs/outputs
    "农业", "产值", "产出", "总产值", "种植", "林业", "牧业", "渔业",
    "耕地", "土地", "亩", "地块", "面积", "流转",
    "农机", "固定资产", "房屋", "役畜",
    "劳动", "工时", "天数", "雇工",
    "种子", "化肥", "农药", "灌溉", "中间品", "消耗",
    # AI index / info tech
    "互联网", "网络", "手机", "智能", "APP", "社交", "咨询", "专家", "政策", "市场", "行情", "价格", "预测", "大数据", "物联网", "无人机",
    # controls
    "年龄", "教育", "受教育", "健康", "培训", "收入", "资产", "合作社",
    # ids
    "pid", "fid", "id",
]


def norm(s: str) -> str:
    if s is None:
        return ""
    if isinstance(s, str):
        return s.strip()
    return str(s).strip()


def hit(text: str) -> bool:
    t = (text or "").lower()
    for kw in KEYWORDS:
        if kw.lower() in t:
            return True
    return False


def safe_read_meta(path: str):
    # Read only metadata. pyreadstat can read metadata without loading all data.
    _df, meta = pyreadstat.read_dta(path, metadataonly=True)
    return meta


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    out_csv = os.path.join(OUTPUT_DIR, "var_catalog.csv")

    files = []
    if os.path.isdir(DATA_DIR):
        for name in os.listdir(DATA_DIR):
            if name.lower().endswith(".dta"):
                files.append(os.path.join(DATA_DIR, name))
    files.sort()

    if not files:
        raise SystemExit(f"No .dta found in {DATA_DIR}")

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file",
            "var_name",
            "var_label",
            "value_labels_keys",
            "matched_keywords",
        ])

        for path in files:
            try:
                meta = safe_read_meta(path)
            except Exception as e:
                writer.writerow([os.path.basename(path), "__ERROR__", str(e), "", ""])
                continue

            # pyreadstat metadata_container fields
            # - column_names: list of variable names
            # - column_names_to_labels: dict var_name -> var_label
            # - variable_to_label: dict var_name -> value-label-set-name (often same as var name)
            # - variable_value_labels: dict var_name -> value-label-set-name
            var_labels = getattr(meta, "column_names_to_labels", None) or {}
            _value_labels_sets = getattr(meta, "value_labels", None) or {}
            var_to_value_label_set = getattr(meta, "variable_value_labels", None) or {}

            for var_name in meta.column_names:
                vlabel = norm(var_labels.get(var_name, ""))

                # value label set name for this var (if any)
                vl_key = ""
                if var_to_value_label_set:
                    vl_key = norm(var_to_value_label_set.get(var_name, ""))

                # match heuristic
                matched = []
                for kw in KEYWORDS:
                    if kw in var_name.lower() or kw in vlabel:
                        matched.append(kw)

                # broader hit based on name/label/value-label-name
                if (not matched) and (hit(var_name) or hit(vlabel) or hit(vl_key)):
                    matched = matched or ["__fuzzy__"]

                # Only output rows with matches to keep file smaller
                if matched:
                    writer.writerow([
                        os.path.basename(path),
                        var_name,
                        vlabel,
                        vl_key,
                        ";".join(sorted(set(matched))),
                    ])

    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
