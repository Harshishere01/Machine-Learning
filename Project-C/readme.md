# Project-C: NFHS-5 Child-Marriage Analysis

This folder contains a reproducible implementation of the **state-grouped cross-validation** experiment for district-level child-marriage prevalence.

## Research target
`Women age 20-24 years married before age 18 years (%)`

## Dataset
Place the NFHS-5 district-level CSV at `datafile.csv`. The supplied dataset in this folder is the original `datafile.csv` provided for this project.

## Setup
```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Run
From this folder:
```bash
python src/grouped_cv.py --data datafile.csv --output results
```

## What the script does
1. Loads the district-level data and normalises column names.
2. Converts numeric columns; values in parentheses retain their numeric value and `*` is treated as missing.
3. Excludes the target, sample-size/count columns, and predictors with more than 40% missing values.
4. Uses **5-fold GroupKFold**, grouping by `State/UT`, so districts from a state cannot appear in both train and test partitions in the same fold.
5. Fits median imputation and standardisation inside each model pipeline and therefore separately within each training fold.
6. Evaluates Ridge Regression, RBF-SVR and Random Forest against a training-fold-mean baseline.
7. Writes fold-level metrics, pooled out-of-fold predictions, summary metrics and an audit JSON file.

## Generated files
- `results/grouped_cv_fold_metrics.csv`
- `results/grouped_cv_results.csv`
- `results/grouped_cv_predictions.csv`
- `results/grouped_cv_audit.json`

## Important interpretation
The reported metrics are based on pooled out-of-fold predictions from unseen states. They are not directly interchangeable with random train/test split metrics. A lower grouped-CV R² is not automatically a defect: predicting districts in entirely unseen states is a harder generalisation task.

## Reproducibility and scope
This script is a clean, executable implementation of the grouped-validation experiment. It is **not claimed to regenerate every figure/table in the submitted paper**. The earlier `code.md`, paper and result files should be retained as historical submitted artifacts until every original analysis has been independently re-run and checked.

