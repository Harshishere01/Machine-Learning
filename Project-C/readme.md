# District-level child-marriage prevalence in India (NFHS-5) - ML Application Assignment, Problem 23

B.Sc. (H) Computer Science, Semester VI, University of Delhi.
Target: % of women aged 20-24 married before age 18, for 706 districts (NFHS-5, 2019-21).
Tasks: regression (prevalence) and classification (district above the all-India 23.3%).

## Repository layout
```
run_experiments.py     # cleaning, models, cross-validation, robustness checks -> results/
build_paper.py         # builds the journal-format PDF from results/results.json
requirements.txt
data/datafile.csv      # NFHS-5 district indicators (put the supplied CSV here)
results/               # created by run_experiments.py (results.json, CSV tables, figures/)
Child_Marriage_ML_Paper.pdf
```

## How to reproduce (about 10 minutes on a laptop)
```
pip install -r requirements.txt
python run_experiments.py            # finds data/datafile.csv automatically, seed 42
python build_paper.py                # writes Child_Marriage_ML_Paper.pdf
```
You can also pass the file explicitly: `python run_experiments.py path/to/datafile.csv`.
Edit the `STUDENT` block at the top of `build_paper.py` for name, college and supervisor.

## What the code does
1. Cleans the text-valued CSV: "(x)" = low-sample estimate (kept), "*" = suppressed (missing); drops columns with more than 40% missing and the survey-size columns.
2. Imputes (median), winsorises (1st-99th percentile) and standardises inside a scikit-learn Pipeline, so nothing leaks from test data.
3. Compares 9 regressors and 8 classifiers (+ a correlation-filter model) against mean and state-mean baselines; 5-fold grid-search CV on the 80% training split; 20% held-out test split.
4. Robustness: state-grouped 5-fold CV (unseen states), leakage check without teenage-motherhood and birth-order indicators, a structural-only set, a focused schooling + wealth model, PCA, permutation importance, bootstrap intervals, per-state errors.

## Main results (test split, seed 42)
- Best regressor: SVR, RMSE 5.18 pp, R2 0.813 (state-mean baseline RMSE 8.36).
- Best classifier: logistic regression, F1 0.862, AUC 0.941.
- Unseen-state (grouped) CV: R2 about 0.70.

## Use of AI tools
[Write exactly what you did yourself and what the AI assistant did: for example, which code was generated, which parts you ran, changed, checked and can explain. Keep it truthful.]

## Data
NFHS-5 (2019-21) district fact sheets, IIPS / MoHFW. See the paper's Data availability section for the source link.
