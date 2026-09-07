```
# ============================================================
# LOAN APPROVAL PREDICTION USING MACHINE LEARNING
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    roc_curve
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

file_path = r"C:\Users\harsh\OneDrive\Desktop\machine learning\07_loan_approval.csv"

df = pd.read_csv(file_path)

print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# ============================================================
# 2. DATA PREPROCESSING
# ============================================================

# ApplicantID is only an identifier.
# It should not be used as a predictive feature.

X = df.drop(
    columns=["ApplicantID", "LoanApproved"]
)

# Convert target:
# N = 0
# Y = 1

y = df["LoanApproved"].map({
    "N": 0,
    "Y": 1
})


# Identify numerical columns

numeric_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


# Identify categorical columns

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()


print("\nNumerical Columns:")
print(numeric_columns)

print("\nCategorical Columns:")
print(categorical_columns)


# ============================================================
# 3. PREPROCESSING PIPELINE
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[

        # Standardize numerical variables
        (
            "num",
            StandardScaler(),
            numeric_columns
        ),

        # One-hot encode categorical variables
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_columns
        )
    ]
)


# ============================================================
# 4. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# ============================================================
# 5. LOGISTIC REGRESSION
# ============================================================

logistic_model = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "model",
        LogisticRegression(
            max_iter=2000,
            random_state=42
        )
    )
])


# Train model

logistic_model.fit(
    X_train,
    y_train
)


# Predictions

y_pred = logistic_model.predict(
    X_test
)


# Probability of approval

y_probability = logistic_model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 6. EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

auc = roc_auc_score(
    y_test,
    y_probability
)


# Regression-style metrics
# These are included only because they
# may be requested in the assignment.

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

mae = mean_absolute_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n================================")
print("LOGISTIC REGRESSION RESULTS")
print("================================")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC-AUC  :", round(auc, 4))

print("\nAdditional Metrics:")
print("MSE  :", round(mse, 4))
print("RMSE :", round(rmse, 4))
print("MAE  :", round(mae, 4))
print("R2   :", round(r2, 4))


# ============================================================
# 7. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Rejected",
            "Approved"
        ]
    )
)


# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title(
    "Logistic Regression Confusion Matrix"
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.xticks(
    [0, 1],
    ["Rejected", "Approved"]
)

plt.yticks(
    [0, 1],
    ["Rejected", "Approved"]
)

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    "loan_confusion_matrix.png",
    dpi=200
)

plt.show()


# ============================================================
# 9. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC={auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve - Loan Approval Prediction"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "loan_roc_curve.png",
    dpi=200
)

plt.show()


# ============================================================
# 10. CLASS DISTRIBUTION GRAPH
# ============================================================

plt.figure(figsize=(7, 5))

df["LoanApproved"].value_counts().sort_index().plot(
    kind="bar"
)

plt.xlabel(
    "Loan Approval"
)

plt.ylabel(
    "Number of Applicants"
)

plt.title(
    "Loan Approval Class Distribution"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    "loan_class_distribution.png",
    dpi=200
)

plt.show()


# ============================================================
# 11. DECISION TREE
# ============================================================

decision_tree = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "model",
        DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        )
    )
])


decision_tree.fit(
    X_train,
    y_train
)

tree_prediction = decision_tree.predict(
    X_test
)

tree_probability = decision_tree.predict_proba(
    X_test
)[:, 1]


tree_accuracy = accuracy_score(
    y_test,
    tree_prediction
)

tree_precision = precision_score(
    y_test,
    tree_prediction
)

tree_recall = recall_score(
    y_test,
    tree_prediction
)

tree_f1 = f1_score(
    y_test,
    tree_prediction
)

tree_auc = roc_auc_score(
    y_test,
    tree_probability
)


# ============================================================
# 12. RANDOM FOREST
# ============================================================

random_forest = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "model",
        RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            random_state=42,
            class_weight="balanced"
        )
    )
])


random_forest.fit(
    X_train,
    y_train
)

rf_prediction = random_forest.predict(
    X_test
)

rf_probability = random_forest.predict_proba(
    X_test
)[:, 1]


rf_accuracy = accuracy_score(
    y_test,
    rf_prediction
)

rf_precision = precision_score(
    y_test,
    rf_prediction
)

rf_recall = recall_score(
    y_test,
    rf_prediction
)

rf_f1 = f1_score(
    y_test,
    rf_prediction
)

rf_auc = roc_auc_score(
    y_test,
    rf_probability
)


# ============================================================
# 13. MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame({

    "Accuracy": [
        accuracy,
        tree_accuracy,
        rf_accuracy
    ],

    "Precision": [
        precision,
        tree_precision,
        rf_precision
    ],

    "Recall": [
        recall,
        tree_recall,
        rf_recall
    ],

    "F1 Score": [
        f1,
        tree_f1,
        rf_f1
    ],

    "ROC-AUC": [
        auc,
        tree_auc,
        rf_auc
    ]

}, index=[
    "Logistic Regression",
    "Decision Tree",
    "Random Forest"
])


print("\n================================")
print("MODEL COMPARISON")
print("================================")

print(
    comparison.round(4)
)


comparison.to_csv(
    "loan_model_comparison.csv"
)


# ============================================================
# 14. PREDICTION TABLE
# ============================================================

prediction_table = X_test.copy()

prediction_table["Actual"] = (
    y_test
    .map({
        0: "Rejected",
        1: "Approved"
    })
)

prediction_table["Predicted"] = (
    pd.Series(
        y_pred,
        index=X_test.index
    )
    .map({
        0: "Rejected",
        1: "Approved"
    })
)

prediction_table["Approval Probability"] = (
    y_probability * 100
)


prediction_table = prediction_table.sort_index()


print("\n================================")
print("PREDICTION TABLE")
print("================================")

print(
    prediction_table.head(20)
)


prediction_table.to_csv(
    "loan_approval_test_predictions.csv",
    index=False
)


# ============================================================
# 15. LOGISTIC REGRESSION COEFFICIENTS
# ============================================================

fitted_preprocessor = (
    logistic_model
    .named_steps["preprocessor"]
)

fitted_model = (
    logistic_model
    .named_steps["model"]
)


feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)


coefficients = (
    fitted_model
    .coef_[0]
)


coefficient_table = pd.DataFrame({

    "Feature": feature_names,

    "Coefficient": coefficients,

    "Odds Ratio": np.exp(coefficients)

})


coefficient_table = (
    coefficient_table
    .sort_values(
        "Coefficient",
        key=abs,
        ascending=False
    )
)


print("\n================================")
print("LOGISTIC REGRESSION COEFFICIENTS")
print("================================")

print(
    coefficient_table
    .round(4)
)


coefficient_table.to_csv(
    "loan_logistic_coefficients.csv",
    index=False
)


# ============================================================
# 16. LOGISTIC REGRESSION EQUATION
# ============================================================

intercept = fitted_model.intercept_[0]

print("\n================================")
print("LOGISTIC REGRESSION EQUATION")
print("================================")

print(
    "log(p/(1-p)) = intercept + Σ(coefficient × feature)"
)

print(
    "Intercept =",
    round(intercept, 6)
)


print("\nModel completed successfully!")
```
## OUTPUT-

<img width="1667" height="962" alt="Image" src="https://github.com/user-attachments/assets/fbcd3dba-7888-456e-8ac1-1ffb47fea32b" />

<img width="572" height="688" alt="Image" src="https://github.com/user-attachments/assets/f785ac27-a56e-407a-b65c-ec950a4786b9" />

<img width="1304" height="912" alt="Image" src="https://github.com/user-attachments/assets/47d70474-7f8b-40d2-8e6a-78553acfaa64" />

<img width="1046" height="933" alt="Image" src="https://github.com/user-attachments/assets/e2f5480b-0329-4419-af3c-128c06494f97" />

<img width="1835" height="937" alt="Image" src="https://github.com/user-attachments/assets/3ce57f47-727e-4516-88c3-6351f5610ee7" />

<img width="1842" height="926" alt="Image" src="https://github.com/user-attachments/assets/4b906bcd-da17-4e2b-a75c-c8149134ce93" />
