# K-Nearest Neighbors (KNN)

```
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

# Step 1: Load dataset
df = pd.read_csv("iris.data", header=None)

df.columns = [
    "SepalLength",
    "SepalWidth",
    "PetalLength",
    "PetalWidth",
    "Species"
]

df.dropna(inplace=True)

print("First five rows:")
print(df.head())

# Step 2: Separate features and target
X = df.iloc[:, 0:4].values
y = df["Species"].values

# Convert species names into numbers
encoder = LabelEncoder()
y = encoder.fit_transform(y)

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

# Step 5: Create KNN model
knn = KNeighborsClassifier(n_neighbors=5)

# Step 6: Train model
knn.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = knn.predict(X_test)

# Step 8: Evaluate model
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=encoder.classes_
))

# Step 9: Predict a new flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]

new_flower = scaler.transform(new_flower)

prediction = knn.predict(new_flower)

print("\nPredicted Flower Species:",
      encoder.inverse_transform(prediction)[0])

# Step 10: Visualize actual vs predicted
plt.figure(figsize=(8, 5))

plt.scatter(
    range(len(y_test)),
    y_test,
    label="Actual",
    marker="o"
)

plt.scatter(
    range(len(y_pred)),
    y_pred,
    label="Predicted",
    marker="x"
)

plt.title("KNN: Actual vs Predicted Classes")
plt.xlabel("Test Sample")
plt.ylabel("Encoded Species")
plt.legend()
plt.show()
```
<img width="674" height="727" alt="image" src="https://github.com/user-attachments/assets/e3c3b077-ea17-4eb4-9709-7025bfc8777f" />
