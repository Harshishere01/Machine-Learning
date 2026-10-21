# CODE
```
"""
run_experiments.py  --  District-level child-marriage prevalence (NFHS-5, 2019-21)
Regenerates every number, table and figure used in the paper.

Usage:  python run_experiments.py
Output: ./results/results.json, ./results/figures/*.png, ./results/*.csv
"""
import json, os, re, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, LogisticRegression, Perceptron)
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
                             confusion_matrix)
from sklearn.model_selection import (train_test_split, GridSearchCV, KFold, StratifiedKFold,
                                     GroupKFold, cross_val_predict, cross_val_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

RAW = os.environ.get("NFHS_CSV", "/mnt/user-data/uploads/datafile.csv")
OUT = "results"; FIG = f"{OUT}/figures"
os.makedirs(FIG, exist_ok=True)
TARGET = "Women age 20-24 years married before age 18 years (%)"
NATIONAL = 23.3          # NFHS-5 all-India figure (%), used as the alert threshold
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.dpi": 220, "font.family": "DejaVu Sans"})

# ------------------------------------------------------------------ data cleaning
def to_num(s):
    s = s.astype(str).str.strip()
    # "(29.5)" = based on 25-49 unweighted cases -> keep value;  "*" = suppressed (<25 cases) -> NaN
    s = s.str.replace(r"^\((.*)\)$", r"\1", regex=True).replace({"*": np.nan, "": np.nan, "nan": np.nan})
    return pd.to_numeric(s, errors="coerce")

df = pd.read_csv("C:\\Users\\harsh\\OneDrive\\Desktop\\machine learning\\ml project\\datafile.csv")
df.columns = [re.sub(r"\s+", " ", c).strip() for c in df.columns]
df["District"] = df["District Names"].str.strip()
df["State"] = df["State/UT"].str.strip().replace({"Maharastra": "Maharashtra"})
all_num = [c for c in df.columns if c not in ("District Names", "State/UT", "District", "State")]
n_paren = int(sum(df_c.astype(str).str.strip().str.match(r"^\(.*\)$").sum() for df_c in
                  [df[c] for c in all_num]))
n_star = int(sum((df[c].astype(str).str.strip() == "*").sum() for c in all_num))
for c in all_num:
    df[c] = to_num(df[c])

miss_pct = (df[all_num].isna().mean() * 100)
dropped_missing = [c for c in all_num if miss_pct[c] > 40]
size_cols = [c for c in all_num if c.startswith("Number of")]
features = [c for c in all_num if c not in dropped_missing + size_cols + [TARGET]]

SHORT_OVERRIDE = {
    "Women age 15-19 years who were already mothers or pregnant at the time of the survey (%)": "Teen mothers/pregnant (15-19)",
    "Women (age 15-49) with 10 or more years of schooling (%)": "Women 10+ yrs schooling",
    "Women (age 15-49) who are literate4 (%)": "Women literate",
    "Population living in households that use an improved sanitation facility2 (%)": "Improved sanitation",
    "Population below age 15 years (%)": "Population <15 yrs",
    "Children under 5 years who are underweight (weight-for-age)18 (%)": "Child underweight",
    "Women (age 15-49 years) whose Body Mass Index (BMI) is below normal (BMI <18.5 kg/m2)21 (%)": "Women BMI < 18.5",
    "Female population age 6 years and above who ever attended school (%)": "Female ever attended school",
    "Women age 15-24 years who use hygienic methods of protection during their menstrual period5 (%)": "Hygienic menstrual protection",
    "Children under 5 years who are stunted (height-for-age)18 (%)": "Child stunting",
    "Women (age 15-49 years) who are overweight or obese (BMI ≥25.0 kg/m2)21 (%)": "Women overweight/obese",
    "Households using clean fuel for cooking3 (%)": "Clean cooking fuel",
    "Pregnant women age 15-49 years who are anaemic (<11.0 g/dl)22 (%)": "Pregnant women anaemic",
    "Prevalence of diarrhoea in the 2 weeks preceding the survey (Children under age 5 years) (%)": "Child diarrhoea",
    "Non-pregnant women age 15-49 years who are anaemic (<12.0 g/dl)22 (%)": "Non-pregnant women anaemic",
    "All women age 15-49 years who are anaemic22 (%)": "All women anaemic",
    "Births in the 5 years preceding the survey that are third or higher order (%)": "Births of order 3+",
    "Men age 15 years and above who use any kind of tobacco (%)": "Men tobacco use",
    "Women age 15 years and above who use any kind of tobacco (%)": "Women tobacco use",
    "Children age 9-35 months who received a vitamin A dose in the last 6 months (%)": "Vitamin A dose (9-35 mo)",
    "Unmet need for spacing (Currently Married Women Age 15-49 years)7 (%)": "Unmet need for spacing",
    "Current users ever told about side effects of current method of family planning8 (%)": "Told of FP side effects",
    "Population living in households with electricity (%)": "Electricity",
    "Deaths in the last 3 years registered with the civil authority (%)": "Death registration",
    "Children under age 5 years whose birth was registered with the civil authority (%)": "Birth registration",
    "Institutional births (in the 5 years before the survey) (%)": "Institutional births",
    "Households using iodized salt (%)": "Iodized salt",
    "Sex ratio of the total population (females per 1,000 males)": "Sex ratio (total)",
    "Mothers who had at least 4 antenatal care visits (for last birth in the 5 years before the survey) (%)": "4+ ANC visits",
    "Current Use of Family Planning Methods (Currently Married Women Age 15-49 years) - Condom (%)": "Condom use",
    "Current Use of Family Planning Methods (Currently Married Women Age 15-49 years) - IUD/PPIUD (%)": "IUD/PPIUD use",
    "Current Use of Family Planning Methods (Currently Married Women Age 15-49 years) - Pill (%)": "Pill use",
}
def short(c):
    if c in SHORT_OVERRIDE: return SHORT_OVERRIDE[c]
    s = re.sub(r"\(%\)|\(Rs\.\)", "", c); s = re.sub(r"(?<=[a-zA-Z\)])\d{1,2}(?:, \d{1,2})?\b", "", s)
    s = re.sub(r"\(for last birth.*?\)|\(in the 5 years before the survey\)", "", s)
    return re.sub(r"\s+", " ", s).strip()[:42]

TEEN = "Women age 15-19 years who were already mothers or pregnant at the time of the survey (%)"
ORD3 = "Births in the 5 years preceding the survey that are third or higher order (%)"
SET_A = features
SET_B = [c for c in features if c not in (TEEN, ORD3)]                       # drop "consequence" indicators
structural_keys = ["Female population age 6", "Population below age 15", "Sex ratio", "birth was registered",
                   "Deaths in the last 3", "electricity", "drinking-water", "sanitation", "clean fuel",
                   "iodized salt", "health insurance", "pre-primary", "who are literate", "10 or more years",
                   "hygienic methods"]
SET_C = [c for c in features if any(k.lower() in c.lower() for k in structural_keys)]  # structural only

y = df[TARGET].values.astype(float)
states = df["State"].values
yc = (y > NATIONAL).astype(int)

# ------------------------------------------------------------------ pipeline pieces
class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lo=1, hi=99): self.lo, self.hi = lo, hi
    def fit(self, X, y=None):
        X = np.asarray(X, float); self.lo_ = np.nanpercentile(X, self.lo, axis=0); self.hi_ = np.nanpercentile(X, self.hi, axis=0); return self
    def transform(self, X): return np.clip(np.asarray(X, float), self.lo_, self.hi_)

def pre(cols, use_state=False):
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("win", Winsorizer()), ("sc", StandardScaler())])
    if not use_state: return ColumnTransformer([("num", num, cols)])
    return ColumnTransformer([("num", num, cols), ("st", OneHotEncoder(handle_unknown="ignore"), ["State"])])

X_all = df[features + ["State"]].copy()

def rmse(a, b): return float(np.sqrt(mean_squared_error(a, b)))
def reg_metrics(a, b): return dict(MAE=float(mean_absolute_error(a, b)), RMSE=rmse(a, b), R2=float(r2_score(a, b)))
def score_fn(m, X):
    return m.decision_function(X) if hasattr(m, "decision_function") else m.predict_proba(X)[:, 1]
def clf_metrics(a, pred, sc):
    return dict(Acc=float(accuracy_score(a, pred)), Prec=float(precision_score(a, pred, zero_division=0)),
                Rec=float(recall_score(a, pred)), F1=float(f1_score(a, pred)),
                AUC=float(roc_auc_score(a, sc)) if sc is not None else None)

# ------------------------------------------------------------------ split
strat = pd.qcut(y, 4, labels=False)
idx = np.arange(len(df))
tr, te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=strat)
Xtr, Xte, ytr, yte = X_all.iloc[tr], X_all.iloc[te], y[tr], y[te]
yctr, ycte = yc[tr], yc[te]
cv_reg = KFold(5, shuffle=True, random_state=SEED)
cv_clf = StratifiedKFold(5, shuffle=True, random_state=SEED)

R = {"meta": {}}
R["meta"].update(n_rows=len(df), n_cols=len(df.columns) - 4 + 2, n_states=int(df.State.nunique()),
                 n_features_all=len(SET_A), n_features_B=len(SET_B), n_features_C=len(SET_C),
                 dropped_missing=[short(c) for c in dropped_missing], n_dropped_missing=len(dropped_missing),
                 miss_dropped_pct=[round(float(miss_pct[c]), 1) for c in dropped_missing],
                 n_size_cols=len(size_cols), n_paren=n_paren, n_star=n_star,
                 n_train=len(tr), n_test=len(te), national=NATIONAL,
                 pos_rate_all=float(yc.mean()), pos_rate_train=float(yctr.mean()), pos_rate_test=float(ycte.mean()),
                 y_mean=float(y.mean()), y_std=float(y.std()), y_median=float(np.median(y)), y_min=float(y.min()),
                 y_max=float(y.max()), n_zero=int((y == 0).sum()),
                 n_dup_names=int(df.District.duplicated(keep=False).sum()),
                 missing_cells_pct=float(df[SET_A].isna().mean().mean() * 100),
                 n_cols_with_missing=int((df[SET_A].isna().sum() > 0).sum()))

# ------------------------------------------------------------------ baselines
def state_mean_pred(train_idx, test_idx):
    g = pd.Series(y[train_idx]).groupby(states[train_idx]).mean()
    return np.array([g.get(s, y[train_idx].mean()) for s in states[test_idx]])
base = {}
base["Mean predictor"] = reg_metrics(yte, np.full_like(yte, ytr.mean()))
base["State-mean predictor"] = reg_metrics(yte, state_mean_pred(tr, te))
R["baseline_reg"] = base
sm_pred = state_mean_pred(tr, te)
bc = {}
bc["Majority class"] = clf_metrics(ycte, np.zeros_like(ycte), np.zeros(len(ycte)))
bc["State-mean rule"] = clf_metrics(ycte, (sm_pred > NATIONAL).astype(int), sm_pred)
R["baseline_clf"] = bc

# ------------------------------------------------------------------ models + grids
def reg_specs(cols, use_state=False):
    P = lambda: pre(cols, use_state)
    return {
        "Linear reg. (OLS)": (Pipeline([("p", P()), ("m", LinearRegression())]), {}),
        "Ridge reg.": (Pipeline([("p", P()), ("m", Ridge())]), {"m__alpha": [0.1, 1, 10, 30, 100, 300, 1000]}),
        "Lasso reg.": (Pipeline([("p", P()), ("m", Lasso(max_iter=20000))]), {"m__alpha": [0.01, 0.03, 0.1, 0.3, 1]}),
        "Polynomial (deg 2) + Ridge": (Pipeline([("p", P()), ("poly", PolynomialFeatures(2, include_bias=False)), ("m", Ridge())]),
                                     {"m__alpha": [1000, 3000]}),
        "Decision tree": (Pipeline([("p", P()), ("m", DecisionTreeRegressor(random_state=SEED))]),
                          {"m__max_depth": [3, 4, 6, 8], "m__min_samples_leaf": [5, 10, 20]}),
        "k-NN regressor": (Pipeline([("p", P()), ("m", KNeighborsRegressor())]),
                           {"m__n_neighbors": [5, 10, 15, 25], "m__weights": ["uniform", "distance"]}),
        "SVR (RBF)": (Pipeline([("p", P()), ("m", SVR(kernel="rbf"))]),
                      {"m__C": [1, 10, 100], "m__epsilon": [1, 3], "m__gamma": ["scale", 0.003, 0.01]}),
        "MLP (64-32)": (Pipeline([("p", P()), ("m", MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=3000, random_state=SEED))]),
                        {"m__alpha": [1e-3, 1e-1, 1, 10]}),
        "Random forest": (Pipeline([("p", P()), ("m", RandomForestRegressor(200, random_state=SEED))]),
                          {"m__max_features": [0.3, "sqrt"], "m__min_samples_leaf": [1, 3]}),
    }

def clf_specs(cols):
    P = lambda: pre(cols)
    return {
        "Logistic regression": (Pipeline([("p", P()), ("m", LogisticRegression(max_iter=5000))]),
                                {"m__C": [0.01, 0.03, 0.1, 0.3, 1, 10], "m__class_weight": [None, "balanced"]}),
        "Gauss. Naive Bayes": (Pipeline([("p", P()), ("m", GaussianNB())]), {"m__var_smoothing": [1e-9, 1e-3, 1e-1, 1]}),
        "Decision tree": (Pipeline([("p", P()), ("m", DecisionTreeClassifier(random_state=SEED))]),
                          {"m__max_depth": [3, 4, 6, 8], "m__min_samples_leaf": [5, 10, 20], "m__class_weight": [None, "balanced"]}),
        "k-NN": (Pipeline([("p", P()), ("m", KNeighborsClassifier())]), {"m__n_neighbors": [5, 10, 15, 25]}),
        "Perceptron": (Pipeline([("p", P()), ("m", Perceptron(penalty="l2", max_iter=3000, random_state=SEED))]),
                       {"m__alpha": [1e-4, 1e-2, 1e-1, 1], "m__class_weight": [None, "balanced"]}),
        "SVM (RBF)": (Pipeline([("p", P()), ("m", SVC(kernel="rbf"))]),
                      {"m__C": [0.1, 1, 10], "m__gamma": ["scale", 0.003, 0.01], "m__class_weight": [None, "balanced"]}),
        "MLP (64-32)": (Pipeline([("p", P()), ("m", MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=3000, random_state=SEED))]),
                        {"m__alpha": [1e-3, 1e-1, 1, 10]}),
        "Random forest": (Pipeline([("p", P()), ("m", RandomForestClassifier(200, random_state=SEED))]),
                          {"m__max_features": [0.3, "sqrt"], "m__min_samples_leaf": [1, 3], "m__class_weight": [None, "balanced"]}),
    }

def pretty_params(p):
    return ", ".join(f"{k.split('__')[1]}={v}" for k, v in p.items()) if p else "-"

# ------------------------------------------------------------------ regression
print("== Regression ==")
reg_res, reg_best, reg_models, reg_preds = {}, {}, {}, {}
for name, (pipe, grid) in reg_specs(SET_A).items():
    if grid:
        gs = GridSearchCV(pipe, grid, cv=cv_reg, scoring="neg_root_mean_squared_error", n_jobs=1).fit(Xtr, ytr)
        model, cvr, bp = gs.best_estimator_, -gs.best_score_, gs.best_params_
    else:
        model = pipe.fit(Xtr, ytr); bp = {}
        cvr = -np.mean(cross_val_score(pipe, Xtr, ytr, cv=cv_reg, scoring="neg_root_mean_squared_error"))
    p = model.predict(Xte)
    reg_res[name] = {**reg_metrics(yte, p), "CV_RMSE": float(cvr)}
    reg_best[name] = bp; reg_models[name] = model; reg_preds[name] = p
    print(f"{name:28s} RMSE={reg_res[name]['RMSE']:.2f} R2={reg_res[name]['R2']:.3f} CV={cvr:.2f}  {bp}")
R["reg"] = reg_res
R["reg_params"] = {k: pretty_params(v) for k, v in reg_best.items()}
best_reg = min(reg_res, key=lambda k: reg_res[k]["RMSE"])
R["best_reg"] = best_reg

# bootstrap CI for test RMSE
rng = np.random.default_rng(SEED)
def boot(fn, B=2000):
    vals = []
    for _ in range(B):
        i = rng.integers(0, len(yte), len(yte)); vals.append(fn(i))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
R["boot"] = {
    "best_rmse_ci": boot(lambda i: rmse(yte[i], reg_preds[best_reg][i])),
    "mean_rmse_ci": boot(lambda i: rmse(yte[i], np.full(len(i), ytr.mean()))),
    "diff_best_vs_state_ci": boot(lambda i: rmse(yte[i], sm_pred[i]) - rmse(yte[i], reg_preds[best_reg][i])),
}

# ------------------------------------------------------------------ classification
print("== Classification ==")
clf_res, clf_best, clf_models, clf_scores, clf_pred = {}, {}, {}, {}, {}
for name, (pipe, grid) in clf_specs(SET_A).items():
    gs = GridSearchCV(pipe, grid, cv=cv_clf, scoring="f1", n_jobs=1).fit(Xtr, yctr)
    m = gs.best_estimator_; sc = score_fn(m, Xte); pr = m.predict(Xte)
    clf_res[name] = {**clf_metrics(ycte, pr, sc), "CV_F1": float(gs.best_score_)}
    clf_best[name] = gs.best_params_; clf_models[name] = m; clf_scores[name] = sc; clf_pred[name] = pr
    print(f"{name:22s} F1={clf_res[name]['F1']:.3f} AUC={clf_res[name]['AUC']:.3f} {gs.best_params_}")
R["clf"] = clf_res
R["clf_params"] = {k: pretty_params(v) for k, v in clf_best.items()}
best_clf = max(clf_res, key=lambda k: clf_res[k]["F1"]); R["best_clf"] = best_clf
tn, fp, fn_, tp = confusion_matrix(ycte, clf_pred[best_clf]).ravel()
R["best_clf_cm"] = dict(TN=int(tn), FP=int(fp), FN=int(fn_), TP=int(tp))
R["boot"]["best_f1_ci"] = boot(lambda i: f1_score(ycte[i], clf_pred[best_clf][i]))
# regression-as-classifier: threshold the best regressor's prediction
rp = reg_preds[best_reg]
R["reg_thresholded"] = clf_metrics(ycte, (rp > NATIONAL).astype(int), rp)

# ------------------------------------------------------------------ state-grouped CV (unseen states)
print("== State-grouped CV ==")
gkf = GroupKFold(5)
folds = list(gkf.split(X_all, y, groups=states))
grp_res, oof = {}, {}
for name in reg_specs(SET_A):
    pipe = clone(reg_models[name])
    p = np.zeros(len(y))
    for a, b in folds:
        p[b] = clone(pipe).fit(X_all.iloc[a], y[a]).predict(X_all.iloc[b])
    oof[name] = p; grp_res[name] = reg_metrics(y, p)
mp = np.zeros(len(y))
for a, b in folds: mp[b] = y[a].mean()
grp_res["Mean predictor"] = reg_metrics(y, mp)
R["grouped"] = grp_res
# random-KFold (same data, same models) for contrast
rnd_res = {}
for name in ["Ridge reg.", "SVR (RBF)", "Random forest"]:
    p = cross_val_predict(clone(reg_models[name]), X_all, y, cv=cv_reg); rnd_res[name] = reg_metrics(y, p)
R["random_cv_all"] = rnd_res
best_grp = min([k for k in grp_res if k != "Mean predictor"], key=lambda k: grp_res[k]["RMSE"]); R["best_grouped"] = best_grp

# per-state error (out-of-state predictions of the best grouped model) vs. global-mean baseline
ps = pd.DataFrame({"State": states, "y": y, "pred": oof[best_grp], "base": mp})
ps["se_m"] = (ps.y - ps.pred) ** 2; ps["se_b"] = (ps.y - ps.base) ** 2
tab = ps.groupby("State").agg(n=("y", "size"), mean_y=("y", "mean"), mean_pred=("pred", "mean"),
                              rmse_m=("se_m", lambda s: np.sqrt(s.mean())), rmse_b=("se_b", lambda s: np.sqrt(s.mean()))).reset_index()
tab["bias"] = tab.mean_pred - tab.mean_y
tab = tab.sort_values("mean_y", ascending=False)
tab.to_csv(f"{OUT}/per_state_error.csv", index=False)
R["per_state"] = tab.round(2).to_dict(orient="records")
R["states_model_beats_baseline"] = int((tab.rmse_m < tab.rmse_b).sum()); R["n_states_tab"] = int(len(tab))
R["states_worse"] = tab[tab.rmse_m >= tab.rmse_b].State.tolist()

# ------------------------------------------------------------------ robustness: feature sets and state indicators
print("== Feature-set robustness ==")
rob = {}
rob_models = ["Ridge reg.", "SVR (RBF)", "MLP (64-32)", "Random forest"]
for setname, cols, us in [("A: all indicators", SET_A, False), ("B: A minus teen-motherhood & birth order", SET_B, False),
                          ("C: structural indicators only", SET_C, False), ("A + state indicators", SET_A, True)]:
    spec = reg_specs(cols, us); rob[setname] = {"n_feat": len(cols) + (int(df.State.nunique()) if us else 0)}
    for name in rob_models:
        pipe = clone(spec[name][0]).set_params(**{k: v for k, v in reg_best[name].items()})
        p = pipe.fit(Xtr, ytr).predict(Xte)
        # also state-grouped CV for set (only meaningful w/o state dummies)
        rob[setname][name] = reg_metrics(yte, p)
        if not us:
            pg = np.zeros(len(y))
            for a, b in folds: pg[b] = clone(pipe).fit(X_all.iloc[a], y[a]).predict(X_all.iloc[b])
            rob[setname][name]["grpRMSE"] = rmse(y, pg); rob[setname][name]["grpR2"] = float(r2_score(y, pg))
R["robust"] = rob

# PCA effect
Z = pre(SET_A).fit(Xtr, ytr).transform(Xtr)
pca = PCA().fit(Z); cum = np.cumsum(pca.explained_variance_ratio_)
R["pca"] = dict(n90=int(np.searchsorted(cum, 0.90) + 1), n95=int(np.searchsorted(cum, 0.95) + 1),
                pc1=float(pca.explained_variance_ratio_[0]), n_feat=int(Z.shape[1]))
pc_res = {}
for lab, steps in {"OLS, all features": [("m", LinearRegression())],
                   "OLS, PCA 95%": [("pca", PCA(0.95)), ("m", LinearRegression())],
                   "k-NN (k=15), all features": [("m", KNeighborsRegressor(15))],
                   "k-NN (k=15), PCA 95%": [("pca", PCA(0.95)), ("m", KNeighborsRegressor(15))]}.items():
    pipe = Pipeline([("p", pre(SET_A))] + steps).fit(Xtr, ytr); pc_res[lab] = reg_metrics(yte, pipe.predict(Xte))
R["pca_effect"] = pc_res

# ------------------------------------------------------------------ interpretation
print("== Interpretation ==")
lasso = reg_models["Lasso reg."]; coef = pd.Series(lasso.named_steps["m"].coef_, index=SET_A)
R["lasso_zero"] = int((coef == 0).sum()); R["lasso_nonzero"] = int((coef != 0).sum())
top = coef.reindex(coef.abs().sort_values(ascending=False).index).head(10)
R["lasso_top"] = [(short(k), round(float(v), 3)) for k, v in top.items()]
ridge = reg_models["Ridge reg."]; rcoef = pd.Series(ridge.named_steps["m"].coef_, index=SET_A)
R["ridge_top"] = [(short(k), round(float(v), 3)) for k, v in rcoef.reindex(rcoef.abs().sort_values(ascending=False).index).head(8).items()]
rf = reg_models["Random forest"]
pi = permutation_importance(rf, Xte, yte, n_repeats=30, random_state=SEED, scoring="neg_root_mean_squared_error")
pimp = pd.Series(pi.importances_mean, index=X_all.columns); pimp = pimp.drop("State", errors="ignore").sort_values(ascending=False)
pstd = pd.Series(pi.importances_std, index=X_all.columns)
R["perm_top"] = [(short(k), round(float(v), 3), round(float(pstd[k]), 3)) for k, v in pimp.head(10).items()]
# same on set B (no teen motherhood / birth order) with RF
rfB = clone(reg_specs(SET_B)["Random forest"][0]).set_params(**reg_best["Random forest"]).fit(Xtr, ytr)
piB = permutation_importance(rfB, Xte, yte, n_repeats=30, random_state=SEED, scoring="neg_root_mean_squared_error")
pB = pd.Series(piB.importances_mean, index=X_all.columns).drop("State", errors="ignore").sort_values(ascending=False)
R["permB_top"] = [(short(k), round(float(v), 3)) for k, v in pB.head(6).items()]
cor_train = df.iloc[tr][SET_A].corrwith(pd.Series(ytr, index=df.index[tr]))
cs = cor_train.sort_values()
R["corr_top_pos"] = [(short(k), round(float(v), 2)) for k, v in cs.iloc[::-1].head(5).items()]
R["corr_top_neg"] = [(short(k), round(float(v), 2)) for k, v in cs.head(5).items()]
R["corr_pairs"] = {
    "teen_vs_school": float(df[[TEEN, "Women (age 15-49) with 10 or more years of schooling (%)"]].corr().iloc[0, 1]),
    "lit_vs_school": float(df[["Women (age 15-49) who are literate4 (%)", "Women (age 15-49) with 10 or more years of schooling (%)"]].corr().iloc[0, 1]),
    "underweight_vs_stunt": float(df[["Children under 5 years who are underweight (weight-for-age)18 (%)", "Children under 5 years who are stunted (height-for-age)18 (%)"]].corr().iloc[0, 1]),
}
# between/within state variance
gm = pd.Series(y).groupby(states).transform("mean")
R["state_r2"] = float(1 - ((y - gm) ** 2).sum() / ((y - y.mean()) ** 2).sum())
st_sum = df.groupby("State")[TARGET].agg(["mean", "size"]).sort_values("mean", ascending=False)
R["state_top"] = [(k, round(float(v["mean"]), 1), int(v["size"])) for k, v in st_sum.head(5).iterrows()]
R["state_bottom"] = [(k, round(float(v["mean"]), 1), int(v["size"])) for k, v in st_sum.tail(5).iterrows()]
R["n_above_nat"] = int((y > NATIONAL).sum())
R["n_above_40"] = int((y > 40).sum())

# ------------------------------------------------------------------ figures
C1, C2, C3 = "#1f4e79", "#c00000", "#e07b00"
# Fig 1 missing values
def lab1(c):
    l = c.lower()
    rules = [("fever or symptoms of ari", "Fever/ARI: taken to provider"), ("non-breastfeeding", "Non-breastfed 6-23 mo: adequate diet"),
             ("6-8 months", "Breastfed 6-8 mo: solid food"),
             ("taken to a health facility or health provider (children under age 5", "Diarrhoea: taken to provider"),
             ("received zinc", "Diarrhoea: received zinc"), ("oral rehydration", "Diarrhoea: received ORS"),
             ("born at home", "Home births: 24-h check-up"), ("exclusively breastfed", "Exclusive breastfeeding"),
             ("private health facility that were delivered", "Private-facility C-section"),
             ("pregnant women age 15-49 years who are anaemic", "Pregnant women anaemic"), ("vaccination card only", "Fully vaccinated (card only)"),
             ("vaccinations in a private", "Vaccinated mostly in private facility"), ("vaccinations in a public", "Vaccinated mostly in public facility")]
    for k, v in rules:
        if k in l: return v
    return short(c)[:34]
m = miss_pct[[c for c in all_num if c != TARGET]].sort_values(ascending=False).head(12)[::-1]
fig, ax = plt.subplots(figsize=(3.4, 3.0)); pos = np.arange(len(m))
ax.barh(pos, m.values, color=[C2 if v > 40 else C1 for v in m.values])
ax.set_yticks(pos); ax.set_yticklabels([lab1(c) for c in m.index], fontsize=5.5)
for i, v in enumerate(m.values): ax.text(v + 1, i, f"{v:.1f}", va="center", fontsize=6)
ax.axvline(40, color="grey", ls="--", lw=0.8); ax.set_xlabel("Missing values (%)"); ax.set_xlim(0, 108)
fig.tight_layout(); fig.savefig(f"{FIG}/fig1_missing.png"); plt.close()

# Fig 2 target distribution + state box plot (single-column, stacked)
fig, axs = plt.subplots(2, 1, figsize=(3.4, 4.6), gridspec_kw={"height_ratios": [1, 1.5]})
axs[0].hist(y, bins=30, color=C1, edgecolor="white"); axs[0].axvline(NATIONAL, color=C2, ls="--", lw=1)
axs[0].text(NATIONAL + 1, axs[0].get_ylim()[1] * 0.88, f"all-India {NATIONAL}%", color=C2, fontsize=6.5)
axs[0].set_xlabel("Women 20-24 married before 18 (%), by district", fontsize=7); axs[0].set_ylabel("Districts", fontsize=7)
big = df.State.value_counts(); keep = big[big >= 8].index
order = df[df.State.isin(keep)].groupby("State")[TARGET].median().sort_values().index
axs[1].boxplot([df[df.State == s][TARGET] for s in order], vert=True, widths=0.6, showfliers=True,
               flierprops=dict(markersize=2), medianprops=dict(color=C2), boxprops=dict(color=C1), whiskerprops=dict(color=C1), capprops=dict(color=C1))
ab = {"Jammu & Kashmir": "J&K", "Andhra Pradesh": "A.P.", "Uttar Pradesh": "U.P.", "Madhya Pradesh": "M.P.", "Himachal Pradesh": "H.P.", "Arunachal Pradesh": "Arun. P.", "Tamil Nadu": "T.N.", "Chhattisgarh": "Chhattis."}
axs[1].set_xticks(range(1, len(order) + 1)); axs[1].set_xticklabels([ab.get(s, s) for s in order], rotation=90, fontsize=5.5)
axs[1].axhline(NATIONAL, color=C2, ls="--", lw=0.8); axs[1].set_ylabel("District prevalence (%)", fontsize=7)
fig.tight_layout(); fig.savefig(f"{FIG}/fig2_target_states.png"); plt.close()

# Fig 3 correlation heatmap of top correlates (training data)
top_cols = list(cs.abs().sort_values(ascending=False).head(11).index)
M = df.iloc[tr][top_cols + [TARGET]].corr()
labs = [short(c)[:30] for c in top_cols] + ["Child marriage (%)"]
fig, ax = plt.subplots(figsize=(3.45, 3.3))
im = ax.imshow(M.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=90, fontsize=4.8); ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=4.8)
for i in range(len(labs)):
    for j in range(len(labs)): ax.text(j, i, f"{M.values[i, j]:.1f}".replace("0.", ".").replace("-.", "-."), ha="center", va="center", fontsize=4, color="white" if abs(M.values[i, j]) > 0.6 else "black")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=5)
fig.tight_layout(); fig.savefig(f"{FIG}/fig3_corr.png"); plt.close()

# Fig 4 PCA
fig, ax = plt.subplots(figsize=(3.4, 2.4)); k = 20
ax.bar(range(1, k + 1), pca.explained_variance_ratio_[:k] * 100, color=C1, label="Individual")
ax.plot(range(1, k + 1), cum[:k] * 100, color=C3, marker="o", ms=3, label="Cumulative"); ax.axhline(95, color="grey", ls=":", lw=0.8)
ax.set_xlabel("Principal component"); ax.set_ylabel("Explained variance (%)"); ax.legend(fontsize=6, frameon=False, loc="center right")
fig.tight_layout(); fig.savefig(f"{FIG}/fig4_pca.png"); plt.close()

# Fig 5 pipeline
steps = [("1  Raw data", "datafile.csv: 706 districts x 109 columns (NFHS-5)"),
         ("2  Cleaning", "'(x)' kept, '*' -> NaN; drop >40% missing, sample-size cols"),
         ("3  Split", "80/20 stratified random split | + state-grouped 5-fold CV"),
         ("4  Pipeline (fit on train)", "median impute, winsorise 1-99%, standardise, [PCA]"),
         ("5  Models + CV", "9 regressors, 8 classifiers, 5-fold grid search"),
         ("6  Evaluation", "MAE, RMSE, R2 | P, R, F1, AUC vs. baselines")]
fig, ax = plt.subplots(figsize=(3.4, 3.1)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(-0.4, 6.2)
for i, (t, s) in enumerate(steps):
    yy = 5.45 - i * 1.05
    ax.add_patch(FancyBboxPatch((0.2, yy - 0.38), 9.6, 0.86, boxstyle="round,pad=0.02", fc="#eef3f8", ec=C1, lw=0.8))
    ax.text(0.45, yy + 0.2, t, fontsize=6.5, weight="bold", va="center"); ax.text(0.45, yy - 0.12, s, fontsize=5.3, va="center", color="#333")
    if i < len(steps) - 1: ax.annotate("", xy=(5, yy - 0.5), xytext=(5, yy - 0.38 - 0.0), arrowprops=dict(arrowstyle="-|>", color=C1, lw=0.8))
fig.tight_layout(); fig.savefig(f"{FIG}/fig5_pipeline.png"); plt.close()

# Fig 6 predicted vs observed
fig, ax = plt.subplots(figsize=(3.4, 3.1))
ax.scatter(yte, reg_preds[best_reg], s=10, alpha=0.65, color=C1, edgecolor="none")
lim = [0, max(60, yte.max() + 3)]; ax.plot(lim, lim, color=C3, lw=1.2); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Observed prevalence (%)"); ax.set_ylabel("Predicted prevalence (%)")
ax.text(2, lim[1] * 0.9, f"{best_reg}\nRMSE = {reg_res[best_reg]['RMSE']:.2f}, R$^2$ = {reg_res[best_reg]['R2']:.3f}", fontsize=6.5)
fig.tight_layout(); fig.savefig(f"{FIG}/fig6_pred_obs.png"); plt.close()

# Fig 7 ROC
fig, ax = plt.subplots(figsize=(3.4, 3.1))
for name, sc in clf_scores.items():
    fpr, tpr, _ = roc_curve(ycte, sc); ax.plot(fpr, tpr, lw=1, label=f"{name} ({clf_res[name]['AUC']:.3f})")
ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=0.8); ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate")
ax.legend(fontsize=5.3, frameon=False, loc="lower right", title="Model (AUC)", title_fontsize=5.5); fig.tight_layout(); fig.savefig(f"{FIG}/fig7_roc.png"); plt.close()

# Fig 8 importance: lasso coef + RF permutation
fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.7))
t10 = coef.reindex(coef.abs().sort_values(ascending=False).index).head(10)[::-1]
axs[0].barh([short(k) for k in t10.index], t10.values, color=[C2 if v > 0 else C1 for v in t10.values]); axs[0].set_xlabel("Lasso coefficient (pp per SD)")
axs[0].tick_params(axis="y", labelsize=6); axs[0].axvline(0, color="k", lw=0.5)
p10 = pimp.head(10)[::-1]
axs[1].barh([short(k) for k in p10.index], p10.values, xerr=pstd[p10.index].values, color=C1, error_kw=dict(lw=0.6)); axs[1].set_xlabel("RMSE increase when permuted (pp)")
axs[1].tick_params(axis="y", labelsize=6); fig.tight_layout(); fig.savefig(f"{FIG}/fig8_importance.png"); plt.close()

# Fig 9 per-state error
t = tab.sort_values("mean_y")
fig, ax = plt.subplots(figsize=(7.1, 2.5)); xx = np.arange(len(t))
ax.bar(xx - 0.2, t.rmse_b, 0.4, color="#9aa7b4", label="Mean-predictor RMSE"); ax.bar(xx + 0.2, t.rmse_m, 0.4, color=C1, label=f"{best_grp} RMSE (unseen-state CV)")
ax.set_xticks(xx); ax.set_xticklabels([s[:16] for s in t.State], rotation=90, fontsize=5); ax.set_ylabel("RMSE (pp)"); ax.legend(fontsize=6, frameon=False, ncol=2)
fig.tight_layout(); fig.savefig(f"{FIG}/fig9_state_error.png"); plt.close()

# save
pd.DataFrame(reg_res).T.to_csv(f"{OUT}/regression_results.csv"); pd.DataFrame(clf_res).T.to_csv(f"{OUT}/classification_results.csv")
with open(f"{OUT}/results.json", "w") as f: json.dump(R, f, indent=1, default=float)
print("\nBest regressor:", best_reg, reg_res[best_reg]); print("Best classifier:", best_clf, clf_res[best_clf])
print("Grouped best:", best_grp, grp_res[best_grp]); print("done")
```

# OUTPUT
<img width="1413" height="525" alt="image" src="https://github.com/user-attachments/assets/43f3a060-172e-4589-8944-fd0cc15822e0" />
