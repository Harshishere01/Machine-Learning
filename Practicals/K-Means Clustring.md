# K-Means Clustering
```
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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

# Step 4: Elbow Method
inertia = []

for k in range(1, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertia.append(model.inertia_)

# Step 5: Plot Elbow graph
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 11),
    inertia,
    marker="o"
)

plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(range(1, 11))
plt.grid(True)

plt.show()

# Step 6: Create K-Means model
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

# Step 7: Train model and assign clusters
labels = kmeans.fit_predict(X_scaled)

# Step 8: Add cluster labels
df["Cluster"] = labels

print("\nCustomers with cluster labels:")
print(df.head(10))

# Step 9: Evaluate clustering
score = silhouette_score(X_scaled, labels)

print("\nSilhouette Score:", score)

# Step 10: Get cluster centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)

print("\nCluster Centroids:")
print(centroids)

# Step 11: Visualize clusters
plt.figure(figsize=(8, 5))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels,
    cmap="viridis"
)

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="X",
    s=200,
    label="Centroids"
)

plt.title("K-Means Customer Segmentation")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()

plt.show()

# Step 12: Display number of customers per cluster
print("\nCustomers in each cluster:")
print(df["Cluster"].value_counts().sort_index())
```
<img width="723" height="668" alt="image" src="https://github.com/user-attachments/assets/343fc0a1-f466-41fa-ba90-8079b09b7fd0" />
<img width="795" height="792" alt="image" src="https://github.com/user-attachments/assets/91b6cc40-8fa8-4f3f-a904-51db5a6cc220" />
