import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import joblib

# Step 1: Load and clean dataset
df = pd.read_csv("kag_risk_factors_cervical_cancer.csv")
df.columns = df.columns.str.strip()
df.replace("?", pd.NA, inplace=True)
df = df.apply(pd.to_numeric, errors="coerce")

# Drop rows where target (Biopsy) is missing
df = df.dropna(subset=["Biopsy"])

# Step 2: Split features and target
X_full = df.drop(columns=["Biopsy"])
y = df["Biopsy"]
numerical_cols = X_full.columns.tolist()

# Step 3: Preprocessing for full data
preprocessor_full = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numerical_cols)
])

X_train_full, _, y_train_full, _ = train_test_split(
    X_full, y, test_size=0.2, random_state=42, stratify=y
)

X_train_processed = preprocessor_full.fit_transform(X_train_full)

# Step 4: Feature importance with full model
rf_temp = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
rf_temp.fit(X_train_processed, y_train_full)

importances = rf_temp.feature_importances_
feat_imp = pd.DataFrame({"Feature": numerical_cols, "Importance": importances})
feat_imp = feat_imp.sort_values(by="Importance", ascending=False)

top_8_features = feat_imp.head(8)["Feature"].tolist()
print("\nTop 8 Important Features:")
print(feat_imp.head(8))

# Plot
feat_imp.head(8).plot(kind="barh", x="Feature", y="Importance", figsize=(8, 5), legend=False)
plt.title("Top 8 Important Features (Biopsy Prediction)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Step 5: Redefine dataset with top 8 features
X = df[top_8_features]

# Step 6: Final preprocessing and split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), top_8_features)
])

# Step 7: Final pipeline with model
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ))
])

# Step 8: Train and evaluate
pipeline.fit(X_train, y_train)
y_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
print("=== First 5 Probabilities ===")
print(y_proba[:5])

# Step 9: Predict on 2 real-time examples (top 8 features only)
realtime_input_low = pd.DataFrame([{
    'Age': 30,
    'Number of sexual partners': 2,
    'First sexual intercourse': 17,
    'STDs:HPV': 1,
    'Dx:HPV': 1,
    'STDs': 0,
    'STDs: Number of diagnosis': 1,
    'STDs: Time since last diagnosis': 1
}], columns=top_8_features)

realtime_input_high = pd.DataFrame([{
    'Age': 45,
    'Number of sexual partners': 7,
    'First sexual intercourse': 14,
    'STDs:HPV': 1,
    'Dx:HPV': 1,
    'STDs': 1,
    'STDs: Number of diagnosis': 3,
    'STDs: Time since last diagnosis': 2
}], columns=top_8_features)

proba_low = pipeline.predict_proba(realtime_input_low)[0][1]
proba_high = pipeline.predict_proba(realtime_input_high)[0][1]

print(f"\n[Low Risk Input] Predicted Biopsy Probability: {proba_low:.2f}")
print(f"[High Risk Input] Predicted Biopsy Probability: {proba_high:.2f}")

# Step 10: Save model
joblib.dump(pipeline, "cervical_cancer_top8_model.joblib")
print("\n✅ Model saved as: cervical_cancer_top8_model.joblib")
