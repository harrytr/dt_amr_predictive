#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_CANDIDATES = [
    Path("."),
    Path("synthetic_endog_import_step2b"),
    Path("synthetic_endog_import_step2"),
]
BASE = next((p for p in BASE_CANDIDATES if p.exists()), BASE_CANDIDATES[0])

OUT_DIR = Path("synthetic_endog_import_step4_figs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAJ_FILE = "h7_trans_majority.csv"
SHARE_FILE = "h7_trans_share.csv"

COHORT_ALIASES = {
    "import_high_train": ["import_high_train", "A_import_majority", "A_import_high"],
    "endog_high_train": ["endog_high_train", "B_endog_majority", "B_endog_high"],
}


def _resolve_cohort_dir(canonical: str) -> tuple[str, Path]:
    aliases = COHORT_ALIASES.get(canonical, [canonical])
    for name in aliases:
        d = BASE / name
        if d.exists():
            return name, d
    return canonical, BASE / canonical


def load_cohort(canonical_name: str) -> pd.DataFrame:
    display_name, cohort_dir = _resolve_cohort_dir(canonical_name)

    if not cohort_dir.exists():
        return pd.DataFrame(columns=["cohort", "sim", "graphml", "majority_label", "share_label"])

    sim_dirs = sorted([p for p in cohort_dir.iterdir() if p.is_dir() and p.name.startswith("sim_")])

    rows = []
    for sd in sim_dirs:
        lab_dir = sd / "labels"
        maj_path = lab_dir / MAJ_FILE
        share_path = lab_dir / SHARE_FILE
        if not maj_path.exists() or not share_path.exists():
            continue

        maj = pd.read_csv(maj_path)
        share = pd.read_csv(share_path)

        if "graphml" not in maj.columns or "label" not in maj.columns:
            continue
        if "graphml" not in share.columns or "label" not in share.columns:
            continue

        maj = maj.rename(columns={"label": "majority_label"})
        share = share.rename(columns={"label": "share_label"})

        df = maj.merge(share, on="graphml", how="inner")
        if df.empty:
            continue

        df["sim"] = sd.name
        df["cohort"] = display_name
        rows.append(df[["cohort", "sim", "graphml", "majority_label", "share_label"]])

    if not rows:
        return pd.DataFrame(columns=["cohort", "sim", "graphml", "majority_label", "share_label"])

    out = pd.concat(rows, ignore_index=True)
    out["majority_label"] = out["majority_label"].astype(int)
    out["share_label"] = out["share_label"].astype(float)
    return out


def main() -> int:
    import_df = load_cohort("import_high_train")
    endog_df = load_cohort("endog_high_train")
    df = pd.concat([import_df, endog_df], ignore_index=True)

    if df.empty:
        raise SystemExit(
            "ERROR: No label rows found. Check that canonical Step 1 sims exist and conversion created sim_*/labels/*.csv.\n"
            f"Looked for BASE in: {', '.join(str(p) for p in BASE_CANDIDATES)} (selected: {BASE}).\n"
            f"Expected label files: {MAJ_FILE} and {SHARE_FILE} under each sim_*/labels/."
        )

    out_csv = OUT_DIR / "mechanism_separation_aggregated_labels.csv"
    df.to_csv(out_csv, index=False)

    cohorts_present = sorted(df["cohort"].unique().tolist())

    plt.figure()
    for cohort in cohorts_present:
        vals = df.loc[df["cohort"] == cohort, "share_label"].replace([math.inf, -math.inf], math.nan).dropna()
        plt.hist(vals.values, bins=20, alpha=0.5, label=cohort)
    plt.axvline(0.5, linestyle="--")
    plt.title("Mechanism separation: h7_trans_share distributions (aggregated over sims)")
    plt.xlabel("h7_trans_share (endogenous transmission share)")
    plt.ylabel("count")
    plt.legend()
    plt.savefig(OUT_DIR / "figure_mechanism_separation_trans_share.png", dpi=250, bbox_inches="tight")
    plt.close()

    rates = df.groupby("cohort")["majority_label"].mean()
    plt.figure()
    plt.bar(rates.index.tolist(), rates.values.tolist())
    plt.ylim(0, 1)
    plt.title("Mechanism separation: P(endogenous majority) by cohort (aggregated over sims)")
    plt.xlabel("cohort")
    plt.ylabel("fraction majority (label=1)")
    plt.savefig(OUT_DIR / "figure_mechanism_separation_majority_rate.png", dpi=250, bbox_inches="tight")
    plt.close()

    print(f"OK: wrote plots + {out_csv} with n={len(df)} rows (BASE={BASE})")
    print("Majority rates:", rates.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
