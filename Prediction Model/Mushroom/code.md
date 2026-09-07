```
# ============================================================
# MUSHROOM EDIBILITY PREDICTION USING MACHINE LEARNING
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder

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
    roc_auc_score,
    roc_curve,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

file_path = r"C:\Users\harsh\OneDrive\Desktop\machine learning\11_mushroom_edibility.csv"

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
# 2. TARGET DISTRIBUTION
# ============================================================

print("\nClass Distribution:")
print(df["Class"].value_counts())


# ============================================================
# 3. DEFINE FEATURES AND TARGET
# ============================================================

# SampleID is only an identifier.
# It should not be used for prediction.

X = df.drop(
    columns=["SampleID", "Class"]
)

# Convert target:
#
# poisonous = 0
# edible = 1

y = df["Class"].map({
    "poisonous": 0,
    "edible": 1
})


# ============================================================
# 4. IDENTIFY CATEGORICAL FEATURES
# ============================================================

categorical_columns = X.columns.tolist()

print("\nCategorical Columns:")
print(categorical_columns)


# ============================================================
# 5. ONE-HOT ENCODING
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore"
            ),

            categorical_columns
        )

    ]
)


# ============================================================
# 6. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\nTraining Samples:")
print(len(X_train))

print("\nTesting Samples:")
print(len(X_test))


# ============================================================
# 7. LOGISTIC REGRESSION
# ============================================================

logistic_model = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "model",

        LogisticRegression(
            max_iter=3000,
            random_state=42
        )
    )

])


# Train model

logistic_model.fit(
    X_train,
    y_train
)


# ============================================================
# 8. PREDICTIONS
# ============================================================

y_pred = logistic_model.predict(
    X_test
)


# Probability of being edible

y_probability = logistic_model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 9. EVALUATION METRICS
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

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


# Additional metrics

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


print("\n==========================================")
print("LOGISTIC REGRESSION RESULTS")
print("==========================================")

print("Accuracy :", round(accuracy, 4))

print("Precision:", round(precision, 4))

print("Recall   :", round(recall, 4))

print("F1 Score :", round(f1, 4))

print("ROC-AUC  :", round(roc_auc, 4))

print("\nAdditional Metrics:")

print("MSE  :", round(mse, 4))

print("RMSE :", round(rmse, 4))

print("MAE  :", round(mae, 4))

print("R²   :", round(r2, 4))


# ============================================================
# 10. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(

        y_test,

        y_pred,

        target_names=[
            "Poisonous",
            "Edible"
        ]

    )
)


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")

print(cm)


plt.figure(
    figsize=(6, 5)
)

plt.imshow(cm)

plt.title(
    "Logistic Regression Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.xticks(
    [0, 1],
    ["Poisonous", "Edible"]
)

plt.yticks(
    [0, 1],
    ["Poisonous", "Edible"]
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
    "mushroom_confusion_matrix.png",
    dpi=200
)

plt.show()


# ============================================================
# 12. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)


plt.figure(
    figsize=(7, 5)
)

plt.plot(
    fpr,
    tpr,

    label=
    f"Logistic Regression "
    f"(AUC = {roc_auc:.3f})"
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
    "ROC Curve - Mushroom Edibility"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "mushroom_roc_curve.png",
    dpi=200
)

plt.show()


# ============================================================
# 13. CLASS DISTRIBUTION GRAPH
# ============================================================

plt.figure(
    figsize=(7, 5)
)

df["Class"].value_counts().sort_index().plot(
    kind="bar"
)

plt.xlabel(
    "Mushroom Class"
)

plt.ylabel(
    "Number of Samples"
)

plt.title(
    "Mushroom Edibility Class Distribution"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    "mushroom_class_distribution.png",
    dpi=200
)

plt.show()


# ============================================================
# 14. DECISION TREE
# ============================================================

decision_tree = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "model",

        DecisionTreeClassifier(
            max_depth=6,
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
# 15. RANDOM FOREST
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

            max_depth=10,

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
# 16. MODEL COMPARISON
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

        roc_auc,
        tree_auc,
        rf_auc

    ]

},

index=[

    "Logistic Regression",

    "Decision Tree",

    "Random Forest"

])


print("\n==========================================")

print("MODEL COMPARISON")

print("==========================================")

print(
    comparison.round(4)
)


comparison.to_csv(
    "mushroom_model_comparison.csv"
)


# ============================================================
# 17. PREDICTION TABLE
# ============================================================

prediction_table = X_test.copy()


prediction_table["Actual Class"] = (

    y_test

    .map({
        0: "Poisonous",
        1: "Edible"
    })

)


prediction_table["Predicted Class"] = (

    pd.Series(
        y_pred,
        index=X_test.index
    )

    .map({
        0: "Poisonous",
        1: "Edible"
    })

)


prediction_table["Edible Probability (%)"] = (

    y_probability * 100

)


prediction_table = (
    prediction_table
    .sort_index()
)


print("\n==========================================")

print("PREDICTION TABLE")

print("==========================================")

print(
    prediction_table.head(20)
)


prediction_table.to_csv(
    "mushroom_test_predictions.csv",
    index=False
)


# ============================================================
# 18. LOGISTIC REGRESSION COEFFICIENTS
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


intercept = (

    fitted_model
    .intercept_[0]

)


coefficient_table = pd.DataFrame({

    "Feature":
        feature_names,

    "Coefficient":
        coefficients,

    "Odds Ratio":
        np.exp(coefficients)

})


coefficient_table = (

    coefficient_table

    .sort_values(
        "Coefficient",
        key=abs,
        ascending=False
    )

)


print("\n==========================================")

print("LOGISTIC REGRESSION COEFFICIENTS")

print("==========================================")

print(
    coefficient_table.round(4)
)


coefficient_table.to_csv(
    "mushroom_logistic_coefficients.csv",
    index=False
)


# ============================================================
# 19. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

rf_preprocessor = (

    random_forest
    .named_steps["preprocessor"]

)


rf_model = (

    random_forest
    .named_steps["model"]

)


rf_feature_names = (

    rf_preprocessor
    .get_feature_names_out()

)


importance = pd.DataFrame({

    "Feature":
        rf_feature_names,

    "Importance":
        rf_model.feature_importances_

})


importance = (

    importance

    .sort_values(
        "Importance",
        ascending=False
    )

)


print("\n==========================================")

print("TOP RANDOM FOREST FEATURES")

print("==========================================")

print(
    importance.head(15).round(4)
)


importance.to_csv(
    "mushroom_random_forest_importance.csv",
    index=False
)


# ============================================================
# 20. FINAL MESSAGE
# ============================================================

print("\n==========================================")

print("MACHINE LEARNING ANALYSIS COMPLETED")

print("==========================================")

print(
    "Best Model: Logistic Regression"
)

print(
    f"Accuracy: {accuracy:.4f}"
)

print(
    f"ROC-AUC: {roc_auc:.4f}"
)
```
## OUTPUT-
  
  <img width="1147" height="947" alt="Image" src="https://github.com/user-attachments/assets/a6dd2065-79e5-4494-b64c-d5623a6ab08c" />

  <img width="617" height="745" alt="Image" src="https://github.com/user-attachments/assets/67e20bc2-a982-4c70-b3d6-53946f7eb1fd" />

  <img width="1225" height="438" alt="Image" src="https://github.com/user-attachments/assets/7beb8dd2-a009-4386-92ab-1780c364fa45" />

  <img width="539" height="720" alt="Image" src="https://github.com/user-attachments/assets/42ba18e9-836b-4f14-965f-624f586162a5" />

  <img width="493" height="476" alt="Image" src="https://github.com/user-attachments/assets/e946f8e3-c6d3-4e54-aaab-632e4fe185fc" />

  <img width="934" height="922" alt="Image" src="https://github.com/user-attachments/assets/022e8d00-1203-4777-8eb3-2fcfeb9ea7bd" />

  <img width="1843" height="939" alt="Image" src="https://github.com/user-attachments/assets/ffec34fd-8645-4072-b489-84f13c6b05d6" />

  <img width="1797" height="920" alt="Image" src="https://github.com/user-attachments/assets/ffb5251b-1c8a-4802-b327-e6293fdd776d" />
