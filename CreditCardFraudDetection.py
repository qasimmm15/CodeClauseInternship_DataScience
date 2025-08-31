# advanced_fraud_detection.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# -------------------------
# 1. Load dataset
# -------------------------
df = pd.read_csv("creditcard.csv")
print("Shape:", df.shape)
print(df['Class'].value_counts())

# -------------------------
# 2. Prepare features / target
# -------------------------
X = df.drop("Class", axis=1)
y = df["Class"]

# The dataset has PCA columns V1..V28 plus Time and Amount.
# Scale Time and Amount (and it's safe to scale all numeric columns for models that benefit).
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # we keep scaler for possible model use later

# -------------------------
# 3. Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)
print("Train:", X_train.shape, "Test:", X_test.shape)

# -------------------------
# 4. Define pipelines
# -------------------------
# SMOTE + classifier pipeline (SMOTE only applied on training folds automatically by pipeline)
rf_pipeline = ImbPipeline(steps=[
    ("smote", SMOTE(random_state=42)),
    ("clf", RandomForestClassifier(random_state=42, n_jobs=-1))
])

xgb_pipeline = ImbPipeline(steps=[
    ("smote", SMOTE(random_state=42)),
    ("clf", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs= -1))
])

# -------------------------
# 5. Hyperparameter search (Randomized) - small search for speed
# -------------------------
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

rf_param_dist = {
    "clf__n_estimators": [100, 200, 400],
    "clf__max_depth": [6, 10, 20, None],
    "clf__min_samples_split": [2, 5, 10],
    "clf__class_weight": [None, "balanced"],
}

xgb_param_dist = {
    "clf__n_estimators": [100, 200, 400],
    "clf__max_depth": [3, 6, 10],
    "clf__learning_rate": [0.01, 0.05, 0.1],
    "clf__subsample": [0.6, 0.8, 1.0],
    "clf__colsample_bytree": [0.6, 0.8, 1.0],
}

# RandomizedSearchCV wrappers
rf_search = RandomizedSearchCV(
    rf_pipeline, rf_param_dist, n_iter=10, scoring="roc_auc", n_jobs=-1, cv=cv, random_state=42, verbose=1
)

xgb_search = RandomizedSearchCV(
    xgb_pipeline, xgb_param_dist, n_iter=10, scoring="roc_auc", n_jobs=-1, cv=cv, random_state=42, verbose=1
)

# -------------------------
# 6. Fit models (RF then XGB)
# -------------------------
print("\nFitting Random Forest (with SMOTE + RandomizedSearchCV)...")
rf_search.fit(X_train, y_train)
print("Best RF params:", rf_search.best_params_)
print("Best RF CV ROC-AUC:", rf_search.best_score_)

print("\nFitting XGBoost (with SMOTE + RandomizedSearchCV)...")
xgb_search.fit(X_train, y_train)
print("Best XGB params:", xgb_search.best_params_)
print("Best XGB CV ROC-AUC:", xgb_search.best_score_)

# -------------------------
# 7. Evaluate on test set
# -------------------------
def evaluate_model(model, X_test, y_test, name="Model"):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    roc_auc = roc_auc_score(y_test, y_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n=== Evaluation: {name} ===")
    print("Accuracy:", acc)
    print("ROC AUC:", roc_auc)
    print("PR AUC:", pr_auc)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')
    plt.plot([0,1],[0,1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Precision-Recall curve
    plt.figure(figsize=(6,5))
    plt.plot(recall, precision, label=f'{name} (PR AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {name}')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Best estimators
best_rf = rf_search.best_estimator_
best_xgb = xgb_search.best_estimator_

evaluate_model(best_rf, X_test, y_test, name="RandomForest (best)")
evaluate_model(best_xgb, X_test, y_test, name="XGBoost (best)")

# -------------------------
# 8. Feature importance (for RandomForest and XGBoost)
# -------------------------
# Note: Because we scaled and passed through pipeline, extract the classifier from pipeline:
rf_clf = best_rf.named_steps["clf"]
xgb_clf = best_xgb.named_steps["clf"]

# If original X column names exist, use them; else create generic names
feature_names = X.columns if hasattr(X, "columns") else [f"f{i}" for i in range(X.shape[1])]

def plot_feature_importance(clf, names, title="Feature importance"):
    try:
        importances = clf.feature_importances_
    except AttributeError:
        print("Model has no feature_importances_ attribute.")
        return
    fi = pd.Series(importances, index=names).sort_values(ascending=False)[:20]
    plt.figure(figsize=(8,5))
    sns.barplot(x=fi.values, y=fi.index)
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

plot_feature_importance(rf_clf, feature_names, "RandomForest Top-20 Feature Importance")
plot_feature_importance(xgb_clf, feature_names, "XGBoost Top-20 Feature Importance")

# -------------------------
# 9. Save best model and scaler
# -------------------------
joblib.dump(best_rf, "best_rf_pipeline.joblib")
joblib.dump(best_xgb, "best_xgb_pipeline.joblib")
joblib.dump(scaler, "scaler.joblib")
print("Saved best models and scaler to disk.")



