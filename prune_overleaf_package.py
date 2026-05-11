#!/usr/bin/env python3
"""
prune_overleaf_package.py

Examples
--------
python prune_overleaf_package.py
python prune_overleaf_package.py --dry-run
python prune_overleaf_package.py --delete-unreferenced-files
python prune_overleaf_package.py --results-root experiments_results
python prune_overleaf_package.py --overleaf-package experiments_results/TRACK_ground_truth/overleaf_package

Purpose
-------
Prunes an overleaf_package so that only the directory branches and, optionally,
files required by the real generated LaTeX code remain.

This version is designed to be run from the AMR parent folder. By default it
looks under:

    ./experiments_results/**/overleaf_package

and, for each discovered overleaf_package, it reads the real generated LaTeX
from one of the following files inside that package:

    latex_snippet.tex
    latex.tex
    figures_snippet.tex
    manuscript_figures.tex
    main.tex
    latex.txt

Behavior
--------
- Reads actual generated LaTeX from disk, not an embedded hard-coded snippet.
- Extracts all \\includegraphics{...} paths from that LaTeX.
- Interprets those paths relative to the overleaf_package root.
- Keeps only directory branches needed by the referenced figures.
- Optionally deletes non-referenced files as well.
- Refuses to delete anything outside the selected overleaf_package.

Notes
-----
- By default, only directories are pruned. Files are kept unless
  --delete-unreferenced-files is provided.
- Missing referenced figure files are reported as warnings.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable, Sequence


LATEX_CANDIDATE_FILENAMES: tuple[str, ...] = (
    "latex_snippet.tex",
    "latex.tex",
    "figures_snippet.tex",
    "manuscript_figures.tex",
    "main.tex",
    "latex.txt",
)


def extract_figure_paths(latex_text: str) -> set[Path]:
    pattern = r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}"
    matches = re.findall(pattern, latex_text)
    return {normalize_relative_path(match) for match in matches}


def normalize_relative_path(rel_path: str) -> Path:
    p = Path(rel_path)
    if p.is_absolute():
        raise ValueError(f"Absolute path not allowed: {rel_path}")

    normalized = Path(*p.parts)
    if ".." in normalized.parts:
        raise ValueError(f"Parent traversal not allowed in figure path: {rel_path}")

    return normalized


def collect_required_dirs(required_files: Iterable[Path]) -> set[Path]:
    required_dirs: set[Path] = set()

    for file_path in required_files:
        current = file_path.parent
        while current != Path(".") and current != current.parent:
            required_dirs.add(current)
            current = current.parent

        if file_path.parts:
            required_dirs.add(Path(file_path.parts[0]))

    return required_dirs


def is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def delete_unneeded_directories(
    root: Path,
    required_dirs: set[Path],
    dry_run: bool = False,
) -> list[Path]:
    deleted: list[Path] = []

    all_dirs = [p for p in root.rglob("*") if p.is_dir()]
    all_dirs.sort(key=lambda p: len(p.parts), reverse=True)

    for abs_dir in all_dirs:
        rel_dir = abs_dir.relative_to(root)

        if rel_dir in required_dirs:
            continue

        if any(parent in required_dirs for parent in rel_dir.parents if parent != Path(".")):
            continue

        if dry_run:
            print(f"[DRY-RUN] Remove directory: {abs_dir}")
        else:
            shutil.rmtree(abs_dir)
            print(f"Removed directory: {abs_dir}")

        deleted.append(abs_dir)

    return deleted


def delete_unreferenced_files(
    root: Path,
    required_files: set[Path],
    latex_file: Path,
    dry_run: bool = False,
) -> list[Path]:
    deleted: list[Path] = []

    protected_names = set(LATEX_CANDIDATE_FILENAMES)

    for abs_file in sorted(p for p in root.rglob("*") if p.is_file()):
        rel_file = abs_file.relative_to(root)

        if abs_file == latex_file:
            continue

        if abs_file.name in protected_names:
            continue

        if rel_file in required_files:
            continue

        if dry_run:
            print(f"[DRY-RUN] Remove file: {abs_file}")
        else:
            abs_file.unlink()
            print(f"Removed file: {abs_file}")

        deleted.append(abs_file)

    return deleted


def read_latex_text(latex_file: Path) -> str:
    try:
        return latex_file.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return latex_file.read_text(encoding="utf-8", errors="replace")


def find_latex_file(overleaf_root: Path, explicit_name: str | None = None) -> Path:
    if explicit_name:
        candidate = overleaf_root / explicit_name
        if not candidate.exists():
            raise FileNotFoundError(
                f"Specified LaTeX file does not exist under overleaf_package: {candidate}"
            )
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Specified LaTeX path is not a file under overleaf_package: {candidate}"
            )
        return candidate

    for filename in LATEX_CANDIDATE_FILENAMES:
        candidate = overleaf_root / filename
        if candidate.exists() and candidate.is_file():
            return candidate

    found_tex_like = sorted(
        p for p in overleaf_root.iterdir() if p.is_file() and p.suffix.lower() in {".tex", ".txt"}
    )
    if len(found_tex_like) == 1:
        return found_tex_like[0]

    searched = ", ".join(LATEX_CANDIDATE_FILENAMES)
    raise FileNotFoundError(
        f"No generated LaTeX file found in {overleaf_root}. "
        f"Tried: {searched}"
    )


def discover_overleaf_packages(results_root: Path) -> list[Path]:
    return sorted(
        p for p in results_root.rglob("overleaf_package")
        if p.exists() and p.is_dir()
    )


def prune_single_overleaf_package(
    root: Path,
    latex_filename: str | None,
    dry_run: bool,
    delete_unreferenced: bool,
) -> int:
    latex_file = find_latex_file(root, explicit_name=latex_filename)
    latex_text = read_latex_text(latex_file)

    required_files = extract_figure_paths(latex_text)
    if not required_files:
        print(
            f"ERROR: No \\includegraphics paths found in LaTeX file: {latex_file}",
            file=sys.stderr,
        )
        return 1

    required_dirs = collect_required_dirs(required_files)

    missing_files: list[Path] = []
    for rel_file in sorted(required_files):
        abs_file = root / rel_file
        if not is_relative_to(abs_file, root):
            print(f"ERROR: Unsafe path outside root detected: {abs_file}", file=sys.stderr)
            return 1
        if not abs_file.exists():
            missing_files.append(rel_file)

    print("=" * 80)
    print(f"Overleaf package: {root}")
    print(f"LaTeX source:      {latex_file}")
    print(f"Referenced figures: {len(required_files)}")
    print(f"Required branches:  {len(required_dirs)}")

    if missing_files:
        print("\nWARNING: These referenced files do not currently exist under this overleaf_package:")
        for rel_file in missing_files:
            print(f"  - {rel_file}")

    deleted_dirs = delete_unneeded_directories(
        root=root,
        required_dirs=required_dirs,
        dry_run=dry_run,
    )

    deleted_files: list[Path] = []
    if delete_unreferenced:
        deleted_files = delete_unreferenced_files(
            root=root,
            required_files=required_files,
            latex_file=latex_file,
            dry_run=dry_run,
        )

    print("\nSummary:")
    print(f"  Directories removed: {len(deleted_dirs)}")
    print(f"  Files removed: {len(deleted_files)}")
    print(f"  Dry run: {'yes' if dry_run else 'no'}")

    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prune generated overleaf_package folders using the real LaTeX code "
            "found inside them."
        )
    )
    parser.add_argument(
        "--results-root",
        default="experiments_results",
        help=(
            "Parent folder from which to discover overleaf_package directories "
            "(default: experiments_results)."
        ),
    )
    parser.add_argument(
        "--overleaf-package",
        default=None,
        help=(
            "Explicit path to a single overleaf_package to prune. If provided, "
            "--results-root discovery is skipped."
        ),
    )
    parser.add_argument(
        "--latex-file",
        default=None,
        help=(
            "Explicit LaTeX filename inside the overleaf_package to parse, "
            "for example latex.txt or main.tex."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without deleting it.",
    )
    parser.add_argument(
        "--delete-unreferenced-files",
        action="store_true",
        help="Also delete files not referenced by the parsed LaTeX.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.overleaf_package:
        roots = [Path.cwd() / args.overleaf_package]
    else:
        results_root = Path.cwd() / args.results_root
        if not results_root.exists():
            print(f"ERROR: Results root not found: {results_root}", file=sys.stderr)
            return 1
        if not results_root.is_dir():
            print(f"ERROR: Results root is not a directory: {results_root}", file=sys.stderr)
            return 1

        roots = discover_overleaf_packages(results_root)
        if not roots:
            print(
                f"ERROR: No overleaf_package directories found under: {results_root}",
                file=sys.stderr,
            )
            return 1

    exit_codes: list[int] = []

    for root in roots:
        if not root.exists():
            print(f"ERROR: Folder not found: {root}", file=sys.stderr)
            exit_codes.append(1)
            continue
        if not root.is_dir():
            print(f"ERROR: Not a directory: {root}", file=sys.stderr)
            exit_codes.append(1)
            continue
        if root.name != "overleaf_package":
            print(
                f"ERROR: Refusing to prune a directory not named 'overleaf_package': {root}",
                file=sys.stderr,
            )
            exit_codes.append(1)
            continue

        try:
            code = prune_single_overleaf_package(
                root=root,
                latex_filename=args.latex_file,
                dry_run=args.dry_run,
                delete_unreferenced=args.delete_unreferenced_files,
            )
        except Exception as exc:
            print(f"ERROR while processing {root}: {exc}", file=sys.stderr)
            code = 1

        exit_codes.append(code)

    return 0 if all(code == 0 for code in exit_codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())