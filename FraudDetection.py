# -------------------------------
# Fraud Detection Project
# -------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# 1. Load dataset
df = pd.read_csv("creditcard.csv")  # Make sure the CSV is in same folder

# 2. Explore data
print("Dataset shape:", df.shape)
print("Fraudulent vs Non-Fraudulent counts:")
print(df['Class'].value_counts())  # 0 = normal, 1 = fraud

# 3. Split features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# 4. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Handle imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
print("After SMOTE:", np.bincount(y_resampled))

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42
)

# 7. Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# 8. Predictions
y_pred = model.predict(X_test)

# 9. Evaluation
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
