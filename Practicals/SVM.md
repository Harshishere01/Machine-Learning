# Support Vector Machine (SVM)
```
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

# Step 1: Load dataset
df = pd.read_csv("wdbc.data", header=None)

# First column is ID
# Second column is diagnosis
# Remaining columns are features

print("First five rows:")
print(df.head())

# Step 2: Separate features and target
X = df.iloc[:, 2:].values

# B = Benign, M = Malignant
y = df.iloc[:, 1].map({
    "B": 0,
    "M": 1
}).values

# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 4: Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Create SVM model
svm = SVC(kernel="linear", C=1.0)

# Step 6: Train model
svm.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = svm.predict(X_test)

# Step 8: Evaluate model
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Benign", "Malignant"]
))

# Step 9: Predict one test sample
sample = X_test[0].reshape(1, -1)

prediction = svm.predict(sample)

if prediction[0] == 0:
    print("\nPredicted Class: Benign")
else:
    print("\nPredicted Class: Malignant")

# Step 10: Visualize confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks([0, 1], ["Benign", "Malignant"])
plt.yticks([0, 1], ["Benign", "Malignant"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.colorbar()
plt.show()
```
<img width="644" height="563" alt="image" src="https://github.com/user-attachments/assets/520674c3-89bd-4a79-bc3b-97cca5db6306" />
<img width="615" height="394" alt="image" src="https://github.com/user-attachments/assets/862931cb-8c47-4d5f-9c09-28b8e628c714" />
