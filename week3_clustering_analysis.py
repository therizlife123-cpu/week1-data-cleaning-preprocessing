# Week 3 — Unsupervised Learning and Clustering Analysis
# Dataset: Iris (public dataset; UCI-origin, distributed by scikit-learn)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

iris = load_iris(as_frame=True)
X = iris.data.copy()
X.columns = ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

ks = range(2, 9)
inertias, silhouettes = [], []
for k in ks:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
print("Cluster sizes:")
print(pd.Series(labels).value_counts().sort_index())
print("\nSilhouette score:", round(silhouette_score(X_scaled, labels), 3))

profile = X.assign(cluster=labels).groupby("cluster").mean().round(2)
print("\nCluster feature means:")
print(profile)

plt.figure(figsize=(8, 5))
plt.plot(list(ks), inertias, marker="o")
plt.axvline(k, linestyle="--")
plt.title("Elbow Method for Selecting Number of Clusters")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Within-cluster sum of squares (inertia)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(list(ks), silhouettes, marker="o")
plt.axvline(k, linestyle="--")
plt.title("Silhouette Score by Number of Clusters")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Average silhouette score")
plt.tight_layout()
plt.show()

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
centers_pca = pca.transform(kmeans.cluster_centers_)

plt.figure(figsize=(8, 5))
for cluster in sorted(set(labels)):
    plt.scatter(X_pca[labels == cluster, 0], X_pca[labels == cluster, 1], label=f"Cluster {cluster}")
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], marker="X", s=160, label="Centroids")
plt.title("K-Means Clusters Visualized with PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.tight_layout()
plt.show()

means = pd.DataFrame(X_scaled, columns=X.columns)
means["cluster"] = labels
means = means.groupby("cluster").mean()
ax = means.T.plot(kind="bar", figsize=(9, 5))
ax.set_title("Standard-Scale Feature Profiles by Cluster")
ax.set_xlabel("Feature")
ax.set_ylabel("Mean standardized value")
plt.xticks(rotation=25, ha="right")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()

# Known species labels are used only after clustering for interpretation, not model fitting.
species = pd.Series(iris.target.map(dict(enumerate(iris.target_names))), name="species")
comparison = pd.crosstab(species, pd.Series(labels, name="cluster"))
print("\nSpecies vs. cluster comparison (post-hoc only):")
print(comparison)
