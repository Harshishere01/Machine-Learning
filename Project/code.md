# CODE
```
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CHILD MARRIAGE MACHINE LEARNING PROJECT
# ============================================================

print("=" * 60)
print("          CHILD MARRIAGE MACHINE LEARNING PROJECT")
print("=" * 60)


# ============================================================
# 1. FIND CSV FILE
# ============================================================

csv_files = glob.glob("*.csv")

print("\nCSV files found:")

for i, file in enumerate(csv_files, 1):
    print(f"{i}. {os.path.basename(file)}")


# ============================================================
# SELECT DATASET
# ============================================================

# We select the CSV which contains the child-marriage data.
# The script checks all CSV files automatically.

df = None
dataset_name = None

for file in csv_files:

    try:
        temp = pd.read_csv(file)

        columns_text = " ".join(
            str(col).lower() for col in temp.columns
        )

        if (
            "married before age 18" in columns_text
            or "child marriage" in columns_text
        ):
            df = temp
            dataset_name = file
            break

    except Exception:
        pass


if df is None:

    print("\nChild-marriage dataset was not automatically detected.")

    # If automatic detection fails, change this filename
    # to your actual dataset filename.
    dataset_name = "datafile.csv"

    df = pd.read_csv(dataset_name)


print("\nUsing dataset:")
print(dataset_name)


# ============================================================
# 2. DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(f"\nRows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# ============================================================
# 3. CLEAN COLUMN NAMES
# ============================================================

# Remove leading/trailing spaces
df.columns = df.columns.astype(str).str.strip()


# ============================================================
# 4. FUNCTION TO FIND COLUMNS FLEXIBLY
# ============================================================

def find_column(columns, keywords):

    """
    Finds a column by checking whether all required keywords
    are present in the column name.

    This avoids errors caused by:
    - extra spaces
    - footnote numbers
    - small naming differences
    """

    for column in columns:

        column_clean = (
            str(column)
            .lower()
            .replace(" ", "")
            .replace("_", "")
        )

        if all(
            keyword.lower().replace(" ", "") in column_clean
            for keyword in keywords
        ):
            return column

    return None


# ============================================================
# 5. REQUIRED VARIABLES
# ============================================================

print("\n" + "=" * 60)
print("VARIABLE CHECK")
print("=" * 60)


# Female education
education_school = find_column(
    df.columns,
    ["female population age 6 years and above", "ever attended school"]
)


# Literacy
literacy = find_column(
    df.columns,
    ["women", "15-49", "literate"]
)


# 10 or more years schooling
schooling_10 = find_column(
    df.columns,
    ["women", "15-49", "10 or more years of schooling"]
)


# Electricity
electricity = find_column(
    df.columns,
    ["population living in households with electricity"]
)


# Drinking water
water = find_column(
    df.columns,
    ["improved drinking-water source"]
)


# Sanitation
sanitation = find_column(
    df.columns,
    ["improved sanitation facility"]
)


# Clean fuel
clean_fuel = find_column(
    df.columns,
    ["households using clean fuel for cooking"]
)


# Health insurance
health_insurance = find_column(
    df.columns,
    ["usual member covered", "health insurance"]
)


# Teenage mothers/pregnancy
teen_mothers = find_column(
    df.columns,
    ["women age 15-19", "already mothers or pregnant"]
)


# Family planning
family_planning = find_column(
    df.columns,
    ["current use of family planning methods",
     "any modern method"]
)


# Target variable
target = find_column(
    df.columns,
    ["women age 20-24 years married before age 18"]
)


# ============================================================
# 6. DISPLAY FOUND VARIABLES
# ============================================================

variables = {
    "Female education": education_school,
    "Literacy": literacy,
    "10+ years schooling": schooling_10,
    "Electricity": electricity,
    "Improved drinking water": water,
    "Improved sanitation": sanitation,
    "Clean cooking fuel": clean_fuel,
    "Health insurance": health_insurance,
    "Teenage mothers/pregnancy": teen_mothers,
    "Family planning": family_planning,
    "Target": target
}


for name, column in variables.items():

    if column is not None:
        print(f"FOUND     : {column}")

    else:
        print(f"NOT FOUND : {name}")


# ============================================================
# 7. CHECK REQUIRED COLUMNS
# ============================================================

features = [
    education_school,
    literacy,
    schooling_10,
    electricity,
    water,
    sanitation,
    clean_fuel,
    health_insurance,
    teen_mothers,
    family_planning
]


if any(column is None for column in features) or target is None:

    print("\nERROR: Some required columns could not be found.")

    print("\nColumns actually present in the dataset:")

    for column in df.columns:
        print("-", column)

    raise SystemExit


# ============================================================
# 8. CREATE MODELING DATASET
# ============================================================

print("\n" + "=" * 60)
print("MODELING DATASET")
print("=" * 60)


model_df = df[features + [target]].copy()

print(f"\nRows before cleaning : {len(model_df)}")
print(f"Columns              : {len(model_df.columns)}")


# ============================================================
# 9. CONVERT VALUES TO NUMERIC
# ============================================================

for column in model_df.columns:

    model_df[column] = pd.to_numeric(
        model_df[column],
        errors="coerce"
    )


# ============================================================
# 10. MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing = model_df.isnull().sum()

for column, value in missing.items():

    print(f"{column}    {value}")


print(f"\nTotal missing values: {missing.sum()}")


# ============================================================
# 11. REMOVE MISSING ROWS
# ============================================================

before_cleaning = len(model_df)

model_df = model_df.dropna()

after_cleaning = len(model_df)

print(f"\nRows removed : {before_cleaning - after_cleaning}")
print(f"Final rows   : {after_cleaning}")
print(f"Final columns: {model_df.shape[1]}")

print("\nRemaining missing values:")
print(model_df.isnull().sum().sum())


# ============================================================
# 12. TARGET STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("TARGET VARIABLE STATISTICS")
print("=" * 60)

print(model_df[target].describe())


# ============================================================
# 13. X AND Y
# ============================================================

X = model_df[features]
y = model_df[target]

print("\n" + "=" * 60)
print("X AND Y INFORMATION")
print("=" * 60)

print(f"\nX shape: {X.shape}")
print(f"Y shape: {y.shape}")


# ============================================================
# 14. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")


# ============================================================
# 15. MODEL TRAINING
# ============================================================

print("\n" + "=" * 60)
print("MODEL TRAINING AND EVALUATION")
print("=" * 60)


# ------------------------------------------------------------
# Linear Regression
# ------------------------------------------------------------

print("\nTraining Linear Regression...")

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)


linear_mae = mean_absolute_error(
    y_test,
    linear_pred
)

linear_mse = mean_squared_error(
    y_test,
    linear_pred
)

linear_rmse = np.sqrt(linear_mse)

linear_r2 = r2_score(
    y_test,
    linear_pred
)


print(f"MAE  : {linear_mae:.4f}")
print(f"MSE  : {linear_mse:.4f}")
print(f"RMSE : {linear_rmse:.4f}")
print(f"R²   : {linear_r2:.4f}")


# ------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------

print("\nTraining Random Forest...")

random_forest = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    max_depth=None
)

random_forest.fit(
    X_train,
    y_train
)

rf_pred = random_forest.predict(X_test)


rf_mae = mean_absolute_error(
    y_test,
    rf_pred
)

rf_mse = mean_squared_error(
    y_test,
    rf_pred
)

rf_rmse = np.sqrt(rf_mse)

rf_r2 = r2_score(
    y_test,
    rf_pred
)


print(f"MAE  : {rf_mae:.4f}")
print(f"MSE  : {rf_mse:.4f}")
print(f"RMSE : {rf_rmse:.4f}")
print(f"R²   : {rf_r2:.4f}")


# ------------------------------------------------------------
# Gradient Boosting
# ------------------------------------------------------------

print("\nTraining Gradient Boosting...")

gradient_model = GradientBoostingRegressor(
    n_estimators=200,
    random_state=42
)

gradient_model.fit(
    X_train,
    y_train
)

gb_pred = gradient_model.predict(X_test)


gb_mae = mean_absolute_error(
    y_test,
    gb_pred
)

gb_mse = mean_squared_error(
    y_test,
    gb_pred
)

gb_rmse = np.sqrt(gb_mse)

gb_r2 = r2_score(
    y_test,
    gb_pred
)


print(f"MAE  : {gb_mae:.4f}")
print(f"MSE  : {gb_mse:.4f}")
print(f"RMSE : {gb_rmse:.4f}")
print(f"R²   : {gb_r2:.4f}")


# ============================================================
# 16. MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting"
    ],

    "MAE": [
        linear_mae,
        rf_mae,
        gb_mae
    ],

    "MSE": [
        linear_mse,
        rf_mse,
        gb_mse
    ],

    "RMSE": [
        linear_rmse,
        rf_rmse,
        gb_rmse
    ],

    "R2 Score": [
        linear_r2,
        rf_r2,
        gb_r2
    ]

})


print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(comparison.to_string(index=False))


# ============================================================
# 17. FIND BEST MODEL
# ============================================================

best_index = comparison["R2 Score"].idxmax()

best_model_name = comparison.loc[
    best_index,
    "Model"
]

best_r2 = comparison.loc[
    best_index,
    "R2 Score"
]


print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(f"\nBest Model : {best_model_name}")
print(f"Best R² Score: {best_r2:.4f}")


# ============================================================
# 18. USE BEST PREDICTIONS
# ============================================================

if best_model_name == "Linear Regression":

    best_predictions = linear_pred

elif best_model_name == "Random Forest":

    best_predictions = rf_pred

else:

    best_predictions = gb_pred


# ============================================================
# 19. ACTUAL VS PREDICTED
# ============================================================

print("\n" + "=" * 60)
print("ACTUAL VS PREDICTED VALUES")
print("=" * 60)


results = pd.DataFrame({

    "Actual": y_test.values,

    "Predicted": best_predictions,

    "Difference": (
        y_test.values - best_predictions
    )

})


print(
    results.head(20).to_string(
        index=False
    )
)


# ============================================================
# 20. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)


if best_model_name == "Random Forest":

    importance = random_forest.feature_importances_

elif best_model_name == "Gradient Boosting":

    importance = gradient_model.feature_importances_

else:

    importance = np.abs(
        linear_model.coef_
    )


importance_df = pd.DataFrame({

    "Feature": features,

    "Importance": importance

})


importance_df = importance_df.sort_values(
    by="Importance",
    ascending=True
)


print(
    importance_df.to_string(
        index=False
    )
)


# ============================================================
# 21. LINEAR REGRESSION COEFFICIENTS
# ============================================================

print("\n" + "=" * 60)
print("LINEAR REGRESSION COEFFICIENTS")
print("=" * 60)


coefficient_df = pd.DataFrame({

    "Feature": features,

    "Coefficient": linear_model.coef_,

    "Absolute Coefficient": np.abs(
        linear_model.coef_
    )

})


coefficient_df = coefficient_df.sort_values(
    by="Absolute Coefficient",
    ascending=False
)


print(
    coefficient_df.to_string(
        index=False
    )
)


# ============================================================
# 22. CREATE FIGURES FOLDER
# ============================================================

os.makedirs(
    "figures",
    exist_ok=True
)


# ============================================================
# 23. ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.scatter(
    y_test,
    best_predictions,
    alpha=0.7
)


minimum = min(
    y_test.min(),
    best_predictions.min()
)

maximum = max(
    y_test.max(),
    best_predictions.max()
)


plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linewidth=2
)


plt.xlabel(
    "Actual Child-Marriage Prevalence (%)"
)

plt.ylabel(
    "Predicted Child-Marriage Prevalence (%)"
)

plt.title(
    f"Actual vs Predicted Values - {best_model_name}"
)

plt.tight_layout()

plt.savefig(
    "figures/actual_vs_predicted.png",
    dpi=300
)

plt.show()


# ============================================================
# 24. FEATURE IMPORTANCE GRAPH
# ============================================================

plt.figure(
    figsize=(10, 7)
)

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel(
    "Feature Importance"
)

plt.ylabel(
    "Predictor Variables"
)

plt.title(
    f"Feature Importance - {best_model_name}"
)

plt.tight_layout()

plt.savefig(
    "figures/feature_importance.png",
    dpi=300
)

plt.show()


# ============================================================
# 25. SAVE CSV FILES
# ============================================================

comparison.to_csv(
    "model_comparison_results.csv",
    index=False
)


results.to_csv(
    "actual_vs_predicted.csv",
    index=False
)


model_df.to_csv(
    "clean_child_marriage_dataset.csv",
    index=False
)


importance_df.to_csv(
    "feature_importance.csv",
    index=False
)


coefficient_df.to_csv(
    "linear_regression_coefficients.csv",
    index=False
)


# ============================================================
# 26. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nFiles created:")

print("1. model_comparison_results.csv")
print("2. actual_vs_predicted.csv")
print("3. clean_child_marriage_dataset.csv")
print("4. actual_vs_predicted.png")
print("5. feature_importance.png")
print("6. linear_regression_coefficients.csv")

print(f"\nBest Model : {best_model_name}")
print(f"Best R² Score: {best_r2:.4f}")

print("\nDone!")
```
# OUTPUT

<img width="1740" height="929" alt="Image" src="https://github.com/user-attachments/assets/e3859aff-7532-4c3b-912e-4f4f795e0e69" />

<img width="1113" height="905" alt="Image" src="https://github.com/user-attachments/assets/416c2249-24ac-456a-8bb7-f1ef6849938e" />

<img width="711" height="625" alt="Image" src="https://github.com/user-attachments/assets/ce7c93c6-c566-42fd-a2cf-77eca855845f" />

<img width="1275" height="923" alt="Image" src="https://github.com/user-attachments/assets/38d64573-31af-4bda-bec3-c79dbecccf6b" />

<img width="533" height="325" alt="Image" src="https://github.com/user-attachments/assets/b2cf5ab9-00d4-482f-8c2c-be320c7dbac6" />

<img width="1187" height="976" alt="Image" src="https://github.com/user-attachments/assets/5a4acee4-0eb5-49ba-88f3-f5ee8c7fe390" />


## model comparison results
<img width="515" height="110" alt="image" src="https://github.com/user-attachments/assets/3c17315c-6165-4efd-860f-01a6acd00744" />

## actual vs predicted
<img width="225" height="840" alt="image" src="https://github.com/user-attachments/assets/de67c9b1-813f-47be-8e31-b6d0c0e42037" />
<img width="219" height="804" alt="image" src="https://github.com/user-attachments/assets/bdeeceb9-085d-4b84-8d0e-32195dcbc106" />
<img width="227" height="822" alt="image" src="https://github.com/user-attachments/assets/b0b17fa4-1e64-4685-9e34-0addbc6285d9" />
<img width="232" height="444" alt="image" src="https://github.com/user-attachments/assets/f7f9cd91-063c-4758-8fae-bcdabd91cf7d" />

<img width="1866" height="940" alt="Image" src="https://github.com/user-attachments/assets/c2e32185-2ad2-4cca-945b-3067f52adb2e" />


## clean child marriage dataset


## linear regression coefficient 
<img width="955" height="252" alt="image" src="https://github.com/user-attachments/assets/2b417392-bf38-4d1d-be86-7af35de9c42b" />

