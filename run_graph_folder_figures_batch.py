#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageOps

IDENTITY = "Harry Triantafyllidis"
TRACKS = ["TRACK_ground_truth", "TRACK_partial_observation"]
OUTPUT_ROOT_NAME = "graph_folder_figures_batch"
LOG_FILENAME = "graph_folder_figures_batch_log.txt"
LATEX_FILENAME = "statistics.tex"
DEFAULT_MAX_GRAPHS = "100%"


class ProgressBar:
    def __init__(self, total: int, prefix: str = "TOTAL") -> None:
        self.total = max(int(total), 1)
        self.current = 0
        self.prefix = prefix

    def show(self, note: str = "") -> None:
        self._render(note)

    def advance(self, note: str = "") -> None:
        self.current = min(self.current + 1, self.total)
        self._render(note)

    def _render(self, note: str = "") -> None:
        width = 32
        filled = int(width * self.current / self.total)
        bar = "#" * filled + "-" * (width - filled)
        msg = f"{self.prefix} [{bar}] {self.current}/{self.total}"
        if note:
            msg += f" | {note}"
        print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run graph_folder_figures.py in compare mode for every kept_graphml step folder "
            "that contains train/ and test/ subfolders, then build summary grids and LaTeX."
        )
    )
    parser.add_argument(
        "--max_graphs",
        type=str,
        default=DEFAULT_MAX_GRAPHS,
        help=(
            'Pass-through value for graph_folder_figures.py --max_graphs. '
            'Supports counts, percentages like "25%%", or "all". Default: 100%%'
        ),
    )
    return parser.parse_args()


def script_root() -> Path:
    return Path(__file__).resolve().parent


def collect_pngs(out_dir: Path) -> List[Path]:
    priority_prefixes = [
        "figure_microgrid_",
        "figure_distributions_",
        "figure_communities_and_centrality_",
        "figure_flow_sankey_",
        "figure_timeline_nodes_edges_",
        "figure_state_percentages_",
        "figure_train_vs_test_shift",
        "figure_train_vs_test_ecdf",
        "figure_timeline_diff_test_minus_train",
    ]
    pngs = sorted([p for p in out_dir.glob("*.png") if p.is_file()], key=lambda p: p.name.lower())

    def key(p: Path) -> tuple[int, str]:
        name = p.name.lower()
        for i, prefix in enumerate(priority_prefixes):
            if name.startswith(prefix):
                return (i, name)
        return (999, name)

    return sorted(pngs, key=key)


def split_pngs_into_groups(pngs: List[Path]) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = [
        {
            "suffix": "part1_structure",
            "title": "Part 1: size, structure, and communities",
            "files": [],
        },
        {
            "suffix": "part2_flow_timeline_state",
            "title": "Part 2: flow, timelines, and state composition",
            "files": [],
        },
        {
            "suffix": "part3_shift",
            "title": "Part 3: train-versus-test shift diagnostics",
            "files": [],
        },
    ]

    for p in pngs:
        name = p.name.lower()
        if (
            name.startswith("figure_microgrid_")
            or name.startswith("figure_distributions_")
            or name.startswith("figure_communities_and_centrality_")
        ):
            groups[0]["files"].append(p)
        elif (
            name.startswith("figure_flow_sankey_")
            or name.startswith("figure_timeline_nodes_edges_")
            or name.startswith("figure_state_percentages_")
        ):
            groups[1]["files"].append(p)
        elif (
            name == "figure_train_vs_test_shift.png"
            or name == "figure_train_vs_test_ecdf.png"
            or name == "figure_timeline_diff_test_minus_train.png"
        ):
            groups[2]["files"].append(p)

    return [g for g in groups if g["files"]]


def make_summary_grid(pngs: List[Path], out_path: Path) -> None:
    if not pngs:
        return

    images = [Image.open(p).convert("RGB") for p in pngs]
    try:
        n = len(images)
        cols = 2 if n > 1 else 1
        rows = math.ceil(n / cols)
        cell_w, cell_h = 1800, 1200
        margin = 80
        grid_w = cols * cell_w + (cols + 1) * margin
        grid_h = rows * cell_h + (rows + 1) * margin
        canvas = Image.new("RGB", (grid_w, grid_h), "white")

        for idx, img in enumerate(images):
            thumb = ImageOps.contain(img, (cell_w, cell_h))
            row, col = divmod(idx, cols)
            x = margin + col * (cell_w + margin) + (cell_w - thumb.width) // 2
            y = margin + row * (cell_h + margin) + (cell_h - thumb.height) // 2
            canvas.paste(thumb, (x, y))

        canvas.save(out_path, dpi=(600, 600))
    finally:
        for img in images:
            img.close()


def _reader(pipe, prefix: str, log_lines: List[str]) -> None:
    try:
        for line in iter(pipe.readline, ""):
            text = line.rstrip()
            if text:
                print(f"[{prefix}] {text}", flush=True)
                log_lines.append(f"[{prefix}] {text}")
    finally:
        pipe.close()


def discover_step_roots(track_dir: Path) -> List[Path]:
    kept_root = track_dir / "kept_graphml"
    if not kept_root.exists():
        return []

    step_roots: List[Path] = []
    for candidate in sorted([p for p in kept_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        train_root = candidate / "train"
        test_root = candidate / "test"
        if train_root.is_dir() and test_root.is_dir():
            step_roots.append(candidate)
    return step_roots


def run_one_step(
    *,
    track_dir: Path,
    step_root: Path,
    max_graphs: str,
    per_step_workers: int,
    log_lines: List[str],
) -> Dict[str, Any]:
    base_dir = script_root()
    driver_script = base_dir / "graph_folder_figures.py"
    track_name = track_dir.name
    step_name = step_root.name
    train_root = step_root / "train"
    test_root = step_root / "test"
    out_root = base_dir / OUTPUT_ROOT_NAME / track_name / step_name
    out_root.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {
        "track": track_name,
        "step": step_name,
        "status": "failed",
        "out_root": out_root,
        "summary_grids": [],
        "note": "",
    }

    if not train_root.exists() or not test_root.exists():
        result["note"] = "Missing train/test subfolders"
        return result

    cmd = [
        sys.executable,
        str(driver_script),
        "--graph_dir",
        str(train_root),
        "--compare_dir",
        str(test_root),
        "--out_dir",
        str(out_root),
        "--identity",
        IDENTITY,
        "--title",
        f"{track_name} {step_name} train vs test",
        "--label",
        "train",
        "--compare_label",
        "test",
        "--workers",
        str(per_step_workers),
        "--max_graphs",
        str(max_graphs),
    ]
    prefix = f"{track_name}/{step_name}"
    log_lines.append(f"=== {prefix} ===")
    log_lines.append(f"RUN: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    t = threading.Thread(target=_reader, args=(proc.stdout, prefix, log_lines), daemon=True)
    t.start()
    rc = proc.wait()
    t.join()

    if rc != 0:
        result["note"] = f"graph_folder_figures.py exited with code {rc}"
        return result

    pngs = collect_pngs(out_root)
    groups = split_pngs_into_groups(pngs)
    summary_grids: List[Dict[str, Any]] = []
    for group in groups:
        summary = out_root / f"{track_name}__{step_name}__summary_grid_{group['suffix']}.png"
        make_summary_grid(group["files"], summary)
        summary_grids.append(
            {
                "path": summary,
                "title": group["title"],
                "suffix": group["suffix"],
                "n_pngs": len(group["files"]),
            }
        )

    result["status"] = "ran"
    result["summary_grids"] = summary_grids
    result["note"] = f"Generated {len(pngs)} PNG(s) across {len(summary_grids)} summary grid(s)"
    return result


def write_statistics_tex(base_dir: Path, results: List[Dict[str, Any]]) -> Path:
    out_path = base_dir / LATEX_FILENAME
    lines: List[str] = []
    lines.append("% Auto-generated by run_graph_folder_figures_batch.py")
    lines.append("% Suggested preamble: \\usepackage{graphicx}")
    lines.append("")

    for res in results:
        lines.append(f"% {res['track']} / {res['step']}")
        if res["status"] == "ran" and res.get("summary_grids"):
            for idx, grid in enumerate(res["summary_grids"], start=1):
                rel = grid["path"].relative_to(base_dir).as_posix()
                lines.append("\\begin{figure}[p]")
                lines.append("  \\centering")
                lines.append(
                    f"  \\includegraphics[width=0.98\\textwidth,height=0.93\\textheight,keepaspectratio]{{{rel}}}"
                )
                lines.append(
                    f"  \\caption{{{res['track'].replace('_', ' ')} {res['step'].replace('_', ' ')}: {grid['title'].lower()}.}}"
                )
                lines.append(
                    f"  \\label{{fig:{res['track'].lower()}_{res['step'].lower()}_part{idx}}}"
                )
                lines.append("\\end{figure}")
                lines.append("")
                if idx != len(res["summary_grids"]):
                    lines.append("\\clearpage")
                    lines.append("")
        else:
            lines.append("\\begin{figure}[p]")
            lines.append("  \\centering")
            lines.append(f"  % Step failed: {res['track']} / {res['step']} | {res['note']}")
            lines.append(
                "  \\caption{Step run failed; see graph\\_folder\\_figures\\_batch\\_log.txt for details.}"
            )
            lines.append("\\end{figure}")
            lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main() -> int:
    args = parse_args()
    base_dir = script_root()
    output_root = base_dir / OUTPUT_ROOT_NAME
    output_root.mkdir(parents=True, exist_ok=True)
    results_root = base_dir / "experiments_results"
    log_path = base_dir / LOG_FILENAME

    track_dirs = [results_root / t for t in TRACKS if (results_root / t).exists()]
    if not track_dirs:
        print(f"ERROR: no track directories found under {results_root}", file=sys.stderr)
        return 1

    step_jobs: List[tuple[Path, Path]] = []
    for track_dir in track_dirs:
        for step_root in discover_step_roots(track_dir):
            step_jobs.append((track_dir, step_root))

    if not step_jobs:
        print(f"ERROR: no kept_graphml step folders with train/test found under {results_root}", file=sys.stderr)
        return 1

    total_cpus = os.cpu_count() or 1
    per_step_workers = max(1, (total_cpus - 1) // 2)
    parallel_jobs = min(max(1, len(step_jobs)), max(1, min(4, total_cpus)))
    log_lines: List[str] = [
        f"Repository root: {base_dir}",
        f"Python executable: {sys.executable}",
        f"Tracks found: {', '.join(p.name for p in track_dirs)}",
        f"Step jobs found: {', '.join(f'{track.name}/{step.name}' for track, step in step_jobs)}",
        f"Parallel step jobs: {parallel_jobs}",
        "",
    ]

    progress = ProgressBar(total=len(step_jobs), prefix="TOTAL")
    progress.show(f"starting | step_jobs={len(step_jobs)} | workers/step={per_step_workers} | max_graphs={args.max_graphs}")

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=parallel_jobs) as ex:
        futs = {
            ex.submit(
                run_one_step,
                track_dir=track_dir,
                step_root=step_root,
                max_graphs=args.max_graphs,
                per_step_workers=per_step_workers,
                log_lines=log_lines,
            ): (track_dir.name, step_root.name)
            for track_dir, step_root in step_jobs
        }
        for fut in as_completed(futs):
            track_name, step_name = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {
                    "track": track_name,
                    "step": step_name,
                    "status": "failed",
                    "summary_grids": [],
                    "note": str(e),
                }
            results.append(res)
            progress.advance(f"{track_name}/{step_name} | {res['status']}")

    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    tex_path = write_statistics_tex(base_dir, sorted(results, key=lambda r: (r["track"], r["step"])))
    print(f"Wrote log: {log_path}")
    print(f"Wrote LaTeX: {tex_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
