# Week 4 — Supervised Learning Model Implementation
# Problem: Multi-class classification of Iris species
# Models: Logistic Regression and Random Forest

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris = load_iris(as_frame=True)
X = iris.data.copy()
X.columns = ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Scaling is inside the pipeline to prevent preprocessing leakage.
logreg = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])
rf = RandomForestClassifier(n_estimators=200, random_state=42)

logreg.fit(X_train, y_train)
rf.fit(X_train, y_train)
pred_lr = logreg.predict(X_test)
pred_rf = rf.predict(X_test)

print("LOGISTIC REGRESSION")
print("Accuracy:", round(accuracy_score(y_test, pred_lr), 4))
print(classification_report(y_test, pred_lr, target_names=iris.target_names))

print("RANDOM FOREST")
print("Accuracy:", round(accuracy_score(y_test, pred_rf), 4))
print(classification_report(y_test, pred_rf, target_names=iris.target_names))

# 5-fold stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr_cv = cross_val_score(logreg, X, y, cv=cv, scoring="accuracy")
rf_cv = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")
print("Logistic Regression mean CV accuracy:", round(lr_cv.mean(), 4))
print("Random Forest mean CV accuracy:", round(rf_cv.mean(), 4))

# Confusion matrices
print("Logistic Regression confusion matrix:\n", confusion_matrix(y_test, pred_lr))
print("Random Forest confusion matrix:\n", confusion_matrix(y_test, pred_rf))

# Random Forest feature importance
importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Random Forest feature importance:\n", importance)

# Cross-validation comparison
plt.figure(figsize=(7, 5))
plt.bar(["Logistic Regression", "Random Forest"], [lr_cv.mean(), rf_cv.mean()])
plt.ylim(0.8, 1.02)
plt.ylabel("Mean 5-fold CV accuracy")
plt.title("Cross-Validation Model Comparison")
plt.tight_layout()
plt.show()
