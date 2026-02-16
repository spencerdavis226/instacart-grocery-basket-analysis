# Instacart Grocery Basket Analysis

Portfolio project analyzing grocery purchasing behavior to identify customer segments and shopping patterns, and translate those findings into actionable recommendations.

## Quick links
- **Executive Summary (PDF):** `report/Instacart Executive Summary.pdf`
- **Notebook — Build dataset:** `notebooks/01_build_analytic_dataset.ipynb`
- **Notebook — Analysis & insights:** `notebooks/02_analysis_and_insights.ipynb`

---

## Project overview
This project turns raw Instacart order/product/customer tables into a reusable analytic dataset, then explores:
- shopping peaks (day-of-week / time-of-day),
- product/category patterns,
- customer behavior signals (e.g., reorder habits),
- and segment differences that support business recommendations.

---

## Repo structure
- `notebooks/`  
  - `01_build_analytic_dataset.ipynb` — reproducible pipeline to create an analytic dataset  
  - `02_analysis_and_insights.ipynb` — feature engineering + EDA + segmentation + findings
- `src/`  
  - Python script versions of the notebooks (same logic, easier to skim)
- `report/`  
  - PDF executive summary
- `assets/`  
  - charts / visuals used in the report
- `tools/`  
  - helper scripts used to generate/organize portfolio artifacts

> Note: `course_notebooks/` contains archived working notebooks from the learning process. The polished deliverables live in `notebooks/` and `src/`.

---

## How to run locally (no data included)
The original Instacart dataset is not included in this public repository.

### 1) Set up environment
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
