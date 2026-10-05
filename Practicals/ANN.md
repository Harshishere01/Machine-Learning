# Artificial Neural Network (ANN)
```
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

# Step 1: Load training and testing datasets
train_df = pd.read_csv("mnist_train.csv")
test_df = pd.read_csv("mnist_test.csv")

print("Training dataset:")
print(train_df.head())

# Step 2: Separate features and labels
X_train = train_df.iloc[:, 1:].values
y_train = train_df.iloc[:, 0].values

X_test = test_df.iloc[:, 1:].values
y_test = test_df.iloc[:, 0].values

# Step 3: Use a smaller subset for faster training
# Remove these lines to use the complete datasets

X_train = X_train[:10000]
y_train = y_train[:10000]

X_test = X_test[:2000]
y_test = y_test[:2000]

# Step 4: Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Step 5: Create ANN model
ann = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    max_iter=30,
    random_state=42,
    early_stopping=True
)

# Step 6: Train the neural network
ann.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = ann.predict(X_test)

# Step 8: Evaluate model
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 9: Display a handwritten digit
index = 0

plt.figure(figsize=(4, 4))

plt.imshow(
    X_test[index].reshape(28, 28),
    cmap="gray"
)

plt.title("Actual Digit: " + str(y_test[index]))
plt.axis("off")
plt.show()

# Step 10: Display predicted digit
print("Actual Digit:", y_test[index])
print("Predicted Digit:", y_pred[index])

# Step 11: Plot training loss
plt.figure(figsize=(8, 5))

plt.plot(ann.loss_curve_)

plt.title("ANN Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
```
<img width="1405" height="711" alt="image" src="https://github.com/user-attachments/assets/8a36f32a-ede3-4975-9c51-9bc32915074b" />
<img width="801" height="689" alt="image" src="https://github.com/user-attachments/assets/743623e6-9a61-40dd-aa9c-b61651004793" />
