# Hierarchical Clustering
```
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Step 1: Load dataset
df = pd.read_csv("Mall_Customers.csv")

print("First five rows:")
print(df.head())

# Step 2: Select features
X = df[[
    "Annual Income (k$)",
    "Spending Score (1-100)"
]].values

# Step 3: Feature scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Step 4: Create hierarchical clustering model
hc = AgglomerativeClustering(
    n_clusters=5,
    linkage="ward"
)

# Step 5: Fit model and assign clusters
labels = hc.fit_predict(X_scaled)

# Step 6: Add cluster labels to dataset
df["Cluster"] = labels

print("\nCustomers with cluster labels:")
print(df.head(10))

# Step 7: Evaluate clustering
score = silhouette_score(X_scaled, labels)

print("\nSilhouette Score:", score)

# Step 8: Visualize clusters
plt.figure(figsize=(8, 5))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels,
    cmap="viridis"
)

plt.title("Hierarchical Clustering")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")

plt.show()

# Step 9: Display number of customers in each cluster
print("\nCustomers in each cluster:")
print(df["Cluster"].value_counts().sort_index())
```
<img width="708" height="455" alt="image" src="https://github.com/user-attachments/assets/9c148d0e-bc53-41d1-a7a9-ae9f96b6b7ad" />
<img width="729" height="527" alt="image" src="https://github.com/user-attachments/assets/40b01f70-c5bc-408c-b206-c35c527d1b3d" />
