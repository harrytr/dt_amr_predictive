#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_CANDIDATES = [
    Path("."),
    Path("synthetic_endog_import_step2b"),
]
BASE = next((p for p in BASE_CANDIDATES if p.exists()), BASE_CANDIDATES[0])

FILES = {
    "trans_share": "h7_trans_share.csv",
    "trans_majority": "h7_trans_majority.csv",
    "import_res": "h7_import_res.csv",
    "trans_res": "h7_trans_res.csv",
    "select_res": "h7_select_res.csv",
}

COHORT_ALIASES = {
    "import_high_train": ["import_high_train", "A_import_majority"],
    "endog_high_train": ["endog_high_train", "B_endog_majority"],
}


def load_label(sim_dir: Path, fname: str) -> pd.DataFrame | None:
    p = sim_dir / "labels" / fname
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "graphml" not in df.columns or "label" not in df.columns:
        return None
    return df[["graphml", "label"]].copy()


def _resolve_cohort_dir(cohort: str) -> tuple[str, Path]:
    for alias in COHORT_ALIASES.get(cohort, [cohort]):
        d = BASE / alias
        if d.exists():
            return alias, d
    return cohort, BASE / cohort


def load_cohort(cohort: str) -> pd.DataFrame:
    display_name, cohort_dir = _resolve_cohort_dir(cohort)
    if not cohort_dir.exists():
        return pd.DataFrame()

    sims = sorted([p for p in cohort_dir.iterdir() if p.is_dir() and p.name.startswith("sim_")])

    rows = []
    for sd in sims:
        dfs = {}
        ok = True
        for k, fn in FILES.items():
            d = load_label(sd, fn)
            if d is None:
                ok = False
                break
            d = d.rename(columns={"label": k})
            dfs[k] = d
        if not ok:
            continue

        df = dfs["trans_share"]
        for k in ["trans_majority", "import_res", "trans_res", "select_res"]:
            df = df.merge(dfs[k], on="graphml", how="inner")

        if df.empty:
            continue

        df["sim"] = sd.name
        df["cohort"] = display_name
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out["trans_share"] = out["trans_share"].astype(float)
    out["trans_majority"] = out["trans_majority"].astype(int)
    for k in ["import_res", "trans_res", "select_res"]:
        out[k] = out[k].astype(float)
    return out


def summarise(df: pd.DataFrame, cohort: str) -> None:
    n = len(df)
    print(f"\n=== {cohort} ===")
    print(f"rows={n}")
    if n == 0:
        return
    print("mean trans_share:", float(df["trans_share"].mean()))
    print("majority rate:", float(df["trans_majority"].mean()))
    print("mean import_res:", float(df["import_res"].mean()))
    print("mean trans_res:", float(df["trans_res"].mean()))
    print("mean select_res:", float(df["select_res"].mean()))
    print("frac import_res == 0:", float((df["import_res"] == 0).mean()))
    print("frac trans_res == 0:", float((df["trans_res"] == 0).mean()))
    print("frac (import_res+trans_res)==0:", float(((df["import_res"] + df["trans_res"]) == 0).mean()))


def main() -> int:
    import_df = load_cohort("import_high_train")
    endog_df = load_cohort("endog_high_train")

    summarise(import_df, "import_high_train")
    summarise(endog_df, "endog_high_train")

    out = pd.concat([import_df, endog_df], ignore_index=True)
    out_path = Path("synthetic_endog_import_step4_figs") / "mechanism_components_h7_step2b.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
