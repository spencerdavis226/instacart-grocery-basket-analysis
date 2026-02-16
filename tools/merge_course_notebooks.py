"""
Merge CareerFoundry-style notebooks into portfolio-ready deliverables.

What it does:
- Reads input .ipynb files from course_notebooks/
- Creates two clean notebooks in notebooks/
- Creates two clean .py scripts in src/
- Drops "course-smell" markdown cells (TOCs, exercises, module headers)
- Drops any cells containing certain unprofessional phrases

Run:
  python tools/merge_course_notebooks.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Tuple

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell


ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "course_notebooks"
OUT_NOTEBOOKS = ROOT / "notebooks"
OUT_SRC = ROOT / "src"

# Files to merge (order matters)
BUILD_DATASET_INPUTS = [
    "4.3 IC Data Import.ipynb",
    "4.5 Data Consistency Checks.ipynb",
    "4.6 Combining & Exporting Data.ipynb",
    "4.6 part 2 Combining & Exporting Data.ipynb",
]

ANALYSIS_INPUTS = [
    "4.7 Deriving New Variables.ipynb",
    "4.8 Grouping Data & Aggregating Variables.ipynb",
]

# Anything containing these phrases will be dropped entirely (any cell type)
DROP_PHRASES = [
    "rabbit hole",
    "gemini pro",
    "exercise",
    "toc",
    "4.",
]

# Markdown filtering: drop course-y markdown cells aggressively
DROP_MD_PATTERNS = [
    r"^\s*#\s*\d+\.\d+",          # headers like "# 4.3 ..."
    r"\bTOC\b",
    r"\bExercise\b",
    r"\bTask\b",
    r"\bAchievement\b",
]


def _contains_drop_phrase(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in DROP_PHRASES)


def _is_course_markdown(md: str) -> bool:
    if not md:
        return True
    for pat in DROP_MD_PATTERNS:
        if re.search(pat, md, flags=re.IGNORECASE | re.MULTILINE):
            return True
    return False


def extract_clean_code_cells(nb_path: Path) -> List[str]:
    """Return a list of cleaned code cell sources from a notebook."""
    nb = nbformat.read(nb_path, as_version=4)
    code_blocks: List[str] = []

    for cell in nb.cells:
        src = cell.get("source", "") or ""
        if _contains_drop_phrase(src):
            continue

        if cell.cell_type == "markdown":
            # drop most markdown from course notebooks
            if _is_course_markdown(src):
                continue
            # If you REALLY want to keep some helpful markdown, you could append it here.
            continue

        if cell.cell_type != "code":
            continue

        stripped = src.strip()
        if not stripped:
            continue

        # Skip trivial cells that are just comments
        if all(line.strip().startswith("#") or not line.strip() for line in stripped.splitlines()):
            continue

        code_blocks.append(stripped)

    return code_blocks


def build_notebook(title: str, overview_md: str, sections: List[Tuple[str, List[str]]]) -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.cells.append(new_markdown_cell(f"# {title}\n\n{overview_md}".strip()))

    for section_title, code_blocks in sections:
        nb.cells.append(new_markdown_cell(f"## {section_title}"))
        for block in code_blocks:
            nb.cells.append(new_code_cell(block))

    return nb


def write_py(title: str, sections: List[Tuple[str, List[str]]], out_path: Path) -> None:
    lines: List[str] = []
    lines.append(f'"""')
    lines.append(f"{title}")
    lines.append("")
    lines.append("Generated from course notebooks via tools/merge_course_notebooks.py")
    lines.append('"""')
    lines.append("")

    for section_title, code_blocks in sections:
        lines.append("")
        lines.append("#" + "-" * 70)
        lines.append(f"# {section_title}")
        lines.append("#" + "-" * 70)
        lines.append("")
        for block in code_blocks:
            lines.append(block)
            lines.append("")

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    OUT_NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    OUT_SRC.mkdir(parents=True, exist_ok=True)

    # --- Build analytic dataset notebook ---
    build_sections: List[Tuple[str, List[str]]] = []

    # You can tweak section mapping if you want, but this is a good default.
    for fname in BUILD_DATASET_INPUTS:
        nb_path = IN_DIR / fname
        if not nb_path.exists():
            raise FileNotFoundError(f"Missing input notebook: {nb_path}")
        code = extract_clean_code_cells(nb_path)
        section_name = fname.replace(".ipynb", "")
        build_sections.append((section_name, code))

    build_nb = build_notebook(
        title="01 — Build Analytic Dataset (Instacart)",
        overview_md=(
            "Creates a reusable analytic dataset from raw Instacart tables.\n\n"
            "**What’s inside:** imports, data quality checks, merges, and export steps.\n\n"
            "**Output:** a single dataset saved locally (CSV/Parquet/Pickle—your choice)."
        ),
        sections=build_sections,
    )

    nbformat.write(build_nb, OUT_NOTEBOOKS / "01_build_analytic_dataset.ipynb")

    write_py(
        title="01 — Build Analytic Dataset (Instacart)",
        sections=build_sections,
        out_path=OUT_SRC / "01_build_analytic_dataset.py",
    )

    # --- Analysis & insights notebook ---
    analysis_sections: List[Tuple[str, List[str]]] = []
    for fname in ANALYSIS_INPUTS:
        nb_path = IN_DIR / fname
        if not nb_path.exists():
            raise FileNotFoundError(f"Missing input notebook: {nb_path}")
        code = extract_clean_code_cells(nb_path)
        section_name = fname.replace(".ipynb", "")
        analysis_sections.append((section_name, code))

    analysis_nb = build_notebook(
        title="02 — Analysis & Insights (Instacart)",
        overview_md=(
            "Explores customer behavior and produces business-facing insights.\n\n"
            "**What’s inside:** feature engineering, aggregation, segmentation, and summary tables.\n\n"
            "Tip: keep charts + final insights in this notebook for hiring managers."
        ),
        sections=analysis_sections,
    )

    nbformat.write(analysis_nb, OUT_NOTEBOOKS / "02_analysis_and_insights.ipynb")

    write_py(
        title="02 — Analysis & Insights (Instacart)",
        sections=analysis_sections,
        out_path=OUT_SRC / "02_analysis_and_insights.py",
    )

    print("✅ Done.")
    print("Created:")
    print(" - notebooks/01_build_analytic_dataset.ipynb")
    print(" - notebooks/02_analysis_and_insights.ipynb")
    print(" - src/01_build_analytic_dataset.py")
    print(" - src/02_analysis_and_insights.py")


if __name__ == "__main__":
    main()
