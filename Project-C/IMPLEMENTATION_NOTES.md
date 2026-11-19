# Implementation notes

## What this package adds
- `src/grouped_cv.py`: executable GroupKFold implementation.
- `notebooks/Project_C_Grouped_CV.ipynb`: executed notebook using the same implementation.
- `data/datafile.csv`: original dataset supplied for this rebuild.
- `results/`: freshly generated fold metrics, pooled out-of-fold predictions and audit metadata.

## Validation performed in this environment
- Dataset loaded successfully: 706 rows and 109 original columns.
- 36 State/UT groups identified after normalising `Maharastra` to `Maharashtra`.
- 97 predictors retained after the documented missingness/count-column/target exclusions.
- Five-fold GroupKFold completed for all three models and the training-fold-mean baseline.
- The script explicitly checks that train and test state sets do not overlap in every fold.
- Imputation and scaling are fitted inside each fold's pipeline.

## Fresh pooled out-of-fold results
| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Ridge | 5.5390 | 6.9946 | 0.6838 |
| RBF-SVR | 6.1059 | 7.7946 | 0.6074 |
| Random Forest | 6.7645 | 8.5019 | 0.5329 |
| Training-fold mean baseline | 10.2331 | 12.4968 | -0.0092 |

These are the results of this new, explicitly documented three-model implementation. They should not be presented as exact reproductions of every table in the original paper unless the remaining experiments are also re-run and checked.
