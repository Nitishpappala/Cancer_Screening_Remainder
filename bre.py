import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
import joblib

# Step 1: Load and clean data
df = pd.read_csv("dataB.csv")
df.columns = df.columns.str.strip()
df.drop(columns=["id"], inplace=True)
df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})

# Step 2: Full feature training for importance calculation
X_full = df.drop("diagnosis", axis=1)
y = df["diagnosis"]
numerical_cols = X_full.columns.tolist()

preprocessor_full = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numerical_cols)
])

X_train_full, _, y_train, _ = train_test_split(
    X_full, y, test_size=0.2, random_state=42, stratify=y
)

rf_temp = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
X_train_processed = preprocessor_full.fit_transform(X_train_full)
rf_temp.fit(X_train_processed, y_train)

# Step 3: Select top 8 features
importances = rf_temp.feature_importances_
feat_imp = pd.DataFrame({"Feature": numerical_cols, "Importance": importances})
feat_imp = feat_imp.sort_values(by="Importance", ascending=False)
top_features = feat_imp.head(8)["Feature"].tolist()

print("\nTop 8 Important Features:")
print(feat_imp.head(8))

# Step 4: Plot top 8
feat_imp.head(8).plot(kind="barh", x="Feature", y="Importance", figsize=(8, 5), legend=False)
plt.title("Top 8 Important Features")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Step 5: Redefine dataset with top features
X = df[top_features]

# Step 6: New train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 7: Define preprocessing and model
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), top_features)
])

base_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    class_weight="balanced"
)

calibrated_clf = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv=5)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", calibrated_clf)
])

# Step 8: Train the model
pipeline.fit(X_train, y_train)

# Step 9: Evaluate
y_proba = pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
print("=== First 5 Test Probabilities ===")
print(y_proba[:5])

# Step 10: Predict on custom examples
# Use only the selected 8 features
example_columns = top_features

high_risk_example = pd.DataFrame([{
    'concavity_worst': 0.65,
    'perimeter_worst': 180.0,
    'concave_points_worst': 0.35,
    'radius_worst': 27.0,
    'radius_mean': 22.5,
    'area_worst': 2500.0,
    'area_mean': 1800.0,
    'perimeter_mean': 150.0
}], columns=example_columns)

low_risk_example = pd.DataFrame([{
    'concavity_worst': 0.04,
    'perimeter_worst': 90.0,
    'concave_points_worst': 0.02,
    'radius_worst': 14.0,
    'radius_mean': 12.0,
    'area_worst': 600.0,
    'area_mean': 450.0,
    'perimeter_mean': 78.0
}], columns=example_columns)

high_proba = pipeline.predict_proba(high_risk_example)[0][1]
low_proba = pipeline.predict_proba(low_risk_example)[0][1]

print(f"\n[High Risk] Predicted malignancy probability: {high_proba:.2f}")
print(f"[Low Risk] Predicted malignancy probability: {low_proba:.2f}")

# Step 11: Save the model
joblib.dump(pipeline, "top8_breast_cancer_model.joblib")
print("\n✅ Model saved as: top8_breast_cancer_model.joblib")
