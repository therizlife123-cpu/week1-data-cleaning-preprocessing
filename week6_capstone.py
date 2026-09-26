# Week 6 — Integrative Capstone Project
# Breast Cancer Diagnostic Analysis: preprocessing, EDA, supervised classification, and clustering

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, silhouette_score

data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
X = df.drop(columns="target")
y = df["target"]

print("Shape:", df.shape)
print("Missing values:", int(df.isna().sum().sum()))
print("Duplicates:", int(df.duplicated().sum()))
print(df.describe())

df["target_name"] = df["target"].map({0:"malignant", 1:"benign"})
print(df["target_name"].value_counts())
print(df.groupby("target_name")[["mean radius","mean texture","mean perimeter","mean area"]].mean())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

logreg = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=3000, random_state=42))
])
rf = RandomForestClassifier(n_estimators=300, random_state=42)

for name, model in [("Logistic Regression", logreg), ("Random Forest", rf)]:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(name)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred))
    print("Recall:", recall_score(y_test, pred))
    print("F1:", f1_score(y_test, pred))
    print(confusion_matrix(y_test, pred))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("LR CV accuracy:", cross_val_score(logreg, X, y, cv=cv).mean())
print("RF CV accuracy:", cross_val_score(rf, X, y, cv=cv).mean())

# Unsupervised analysis
Xs = StandardScaler().fit_transform(X)
scores = []
for k in range(2, 8):
    labels = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(Xs)
    scores.append((k, silhouette_score(Xs, labels)))
print("Silhouette scores:", scores)
best_k = max(scores, key=lambda x: x[1])[0]
labels = KMeans(n_clusters=best_k, n_init=20, random_state=42).fit_predict(Xs)
Z = PCA(n_components=2, random_state=42).fit_transform(Xs)
print("Selected k:", best_k)
