#!/usr/bin/env python3
import os
import glob
import csv
import torch


def get_scalar(attr):
    if attr is None:
        return None
    if hasattr(attr, "detach"):
        attr = attr.detach().cpu()
    try:
        return float(attr.view(-1)[0].item())
    except Exception:
        return None


def main():
    graph_dir = "synthetic_amr_graphs_train"
    pt_files = sorted(glob.glob(os.path.join(graph_dir, "*.pt")))
    if not pt_files:
        raise RuntimeError(f"No .pt files found in: {graph_dir}")

    out_csv = os.path.join(graph_dir, "audit_pt_endog_import_h7.csv")
    rows = []
    bad_share = 0
    bad_maj = 0
    missing = 0

    for p in pt_files:
        data = torch.load(p, map_location="cpu")

        tr = get_scalar(getattr(data, "y_h7_trans_res", None))
        sel = get_scalar(getattr(data, "y_h7_select_res", None))
        imp = get_scalar(getattr(data, "y_h7_import_res", None))
        share_stored = get_scalar(getattr(data, "y_h7_endog_share", None))
        maj_stored = get_scalar(getattr(data, "y_h7_endog_majority", None))

        fname = getattr(data, "filename", os.path.basename(p))

        if tr is None or sel is None or imp is None or share_stored is None or maj_stored is None:
            missing += 1
            rows.append([fname, tr, sel, imp, None, share_stored, None, maj_stored, None])
            continue

        endog = tr + sel
        denom = endog + imp
        share = 0.0 if denom <= 0.0 else (endog / denom)
        maj = 1.0 if share >= 0.5 else 0.0

        share_diff = abs(share - share_stored)
        maj_diff = abs(maj - maj_stored)

        if share_diff > 1e-9:
            bad_share += 1
        if maj_diff > 0.0:
            bad_maj += 1

        rows.append([fname, tr, sel, imp, share, share_stored, maj, maj_stored, share_diff])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "pt_filename",
            "y_h7_trans_res",
            "y_h7_select_res",
            "y_h7_import_res",
            "share_recomputed",
            "y_h7_endog_share",
            "maj_recomputed",
            "y_h7_endog_majority",
            "abs_share_diff",
        ])
        w.writerows(rows)

    print(f"Wrote: {out_csv}")
    print(f"PT files: {len(pt_files)} | missing fields: {missing}")
    print(f"Share mismatches: {bad_share} | Majority mismatches: {bad_maj}")

    if bad_share > 0 or bad_maj > 0:
        raise SystemExit("Audit FAILED: stored labels are internally inconsistent.")
    print("Audit PASSED.")


if __name__ == "__main__":
    main()
