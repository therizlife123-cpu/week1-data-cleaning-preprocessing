# Week 2 — Exploratory Data Analysis and Visualization
# Dataset: Iris (UCI Machine Learning Repository / scikit-learn copy)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.frame.copy()
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
df = df.drop(columns=["target"])
df = df.rename(columns={
    "sepal length (cm)": "sepal_length_cm",
    "sepal width (cm)": "sepal_width_cm",
    "petal length (cm)": "petal_length_cm",
    "petal width (cm)": "petal_width_cm"
})

print("Shape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isna().sum())
print("\nDescriptive statistics:\n", df.describe())
print("\nClass distribution:\n", df["species"].value_counts())

# Grouped summary
summary = (df.groupby("species")
             .agg(mean_sepal_length=("sepal_length_cm", "mean"),
                  mean_sepal_width=("sepal_width_cm", "mean"),
                  mean_petal_length=("petal_length_cm", "mean"),
                  mean_petal_width=("petal_width_cm", "mean"))
             .round(2))
print("\nGrouped means:\n", summary)

# Scatter plots
for x, y, title in [
    ("sepal_length_cm", "sepal_width_cm", "Sepal Length vs Sepal Width by Species"),
    ("petal_length_cm", "petal_width_cm", "Petal Length vs Petal Width by Species")
]:
    plt.figure(figsize=(8, 5))
    for species in iris.target_names:
        subset = df[df["species"] == species]
        plt.scatter(subset[x], subset[y], label=species)
    plt.title(title)
    plt.xlabel(x.replace("_", " ").title() + " (cm)")
    plt.ylabel(y.replace("_", " ").title() + " (cm)")
    plt.legend(title="Species")
    plt.tight_layout()
    plt.show()

# Grouped means
means = df.groupby("species")[[
    "sepal_length_cm", "sepal_width_cm",
    "petal_length_cm", "petal_width_cm"
]].mean()
ax = means.plot(kind="bar", figsize=(9, 5))
ax.set_title("Mean Measurements by Species")
ax.set_xlabel("Species")
ax.set_ylabel("Mean measurement (cm)")
ax.legend(title="Feature")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Box plot
plt.figure(figsize=(8, 5))
df.boxplot(column="petal_length_cm", by="species")
plt.title("Petal Length Distribution by Species")
plt.suptitle("")
plt.xlabel("Species")
plt.ylabel("Petal length (cm)")
plt.tight_layout()
plt.show()

# Correlation matrix
numeric_cols = ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"]
corr = df[numeric_cols].corr()
print("\nCorrelation matrix:\n", corr.round(2))

plt.figure(figsize=(7, 6))
plt.imshow(corr, aspect="auto")
plt.colorbar(label="Pearson correlation")
plt.xticks(range(4), numeric_cols, rotation=35, ha="right")
plt.yticks(range(4), numeric_cols)
for i in range(4):
    for j in range(4):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
plt.title("Correlation Matrix of Iris Measurements")
plt.tight_layout()
plt.show()

# Distribution
plt.figure(figsize=(8, 5))
for species in iris.target_names:
    subset = df[df["species"] == species]
    plt.hist(subset["petal_length_cm"], alpha=0.55, label=species, bins=8)
plt.title("Petal Length Distribution by Species")
plt.xlabel("Petal length (cm)")
plt.ylabel("Frequency")
plt.legend(title="Species")
plt.tight_layout()
plt.show()

# IQR anomaly screening
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    count = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f"{col}: {count} IQR-flagged observations")

print("\nEDA completed.")
