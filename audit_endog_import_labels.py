#!/usr/bin/env python3
import os
import glob
import csv
import networkx as nx


def parse_prefix_day(fname: str):
    base = os.path.basename(fname)
    if "_t" not in base:
        return None
    prefix = base.split("_t")[0]
    day_part = base.split("_t")[1].split(".")[0]
    try:
        t = int(day_part)
    except Exception:
        return None
    return prefix, t


def load_label_csv(path: str):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", newline="") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) < 2:
                continue
            out[row[0]] = float(row[1])
    return out


def main():
    graph_dir = "synthetic_amr_audit"
    label_dir = os.path.join(graph_dir, "labels")

    files = sorted(glob.glob(os.path.join(graph_dir, "*.graphml")))
    if not files:
        raise RuntimeError(f"No .graphml files found in {graph_dir}")

    h7_share = load_label_csv(os.path.join(label_dir, "h7_endog_share.csv"))
    h7_maj = load_label_csv(os.path.join(label_dir, "h7_endog_majority.csv"))

    groups = {}
    for f in files:
        parsed = parse_prefix_day(f)
        if parsed is None:
            continue
        prefix, t = parsed
        groups.setdefault(prefix, []).append((t, f))

    out_rows = []
    for prefix, seq in groups.items():
        seq.sort(key=lambda z: z[0])
        day_to_file = {t: f for t, f in seq}
        days = [t for t, _ in seq]

        for t in days:
            H = 7
            trans_sum = 0
            select_sum = 0
            import_sum = 0
            ok = True
            for k in range(1, H + 1):
                tn = t + k
                if tn not in day_to_file:
                    ok = False
                    break
                G = nx.read_graphml(day_to_file[tn])
                trans_sum += int(G.graph.get("new_trans_cr_total", 0))
                select_sum += int(G.graph.get("new_select_cr_total", 0))
                import_sum += int(G.graph.get("new_import_cr_total", 0)) + int(G.graph.get("new_import_ir_total", 0))

            if not ok:
                continue

            endog_sum = trans_sum + select_sum
            denom = endog_sum + import_sum
            share = 0.0 if denom <= 0 else float(endog_sum) / float(denom)
            maj = 1.0 if share >= 0.5 else 0.0

            fname_key = os.path.basename(day_to_file[t])
            share_ref = h7_share.get(fname_key, None)
            maj_ref = h7_maj.get(fname_key, None)

            out_rows.append([
                prefix, t, trans_sum, select_sum, import_sum, share, maj,
                "" if share_ref is None else share_ref,
                "" if maj_ref is None else maj_ref,
                "" if share_ref is None else abs(share - float(share_ref)),
                "" if maj_ref is None else abs(maj - float(maj_ref)),
            ])

    out_path = os.path.join(graph_dir, "audit_endog_import_h7.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "sim_prefix", "t",
            "H7_trans_res", "H7_select_res", "H7_import_res",
            "share_recomputed", "maj_recomputed",
            "share_convert_to_pt", "maj_convert_to_pt",
            "abs_share_diff", "abs_maj_diff"
        ])
        w.writerows(out_rows)

    print(f"Wrote: {out_path}")
    bad_share = [r for r in out_rows if r[9] != "" and float(r[9]) > 1e-9]
    bad_maj = [r for r in out_rows if r[10] != "" and float(r[10]) > 0.0]
    print(f"Share mismatches: {len(bad_share)}; Majority mismatches: {len(bad_maj)}")
    if bad_share or bad_maj:
        raise SystemExit("Audit failed: labels do not match recomputation.")
    print("Audit passed.")


if __name__ == "__main__":
    main()
