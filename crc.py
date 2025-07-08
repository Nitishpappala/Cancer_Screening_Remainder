import pandas as pd

# Step 1: Load and clean the dataset
df = pd.read_csv("crc_dataset.csv")
df.columns = df.columns.str.strip()  # Remove any extra spaces

# Show column names and null counts
print(df.columns.tolist())
print("Missing values:\n", df.isnull().sum())

# Fill nulls in 'Pre-existing Conditions' with "None"
df["Pre-existing Conditions"] = df["Pre-existing Conditions"].fillna("None")
print("Missing values after fill:\n", df.isnull().sum())

# Drop ID column (make sure no trailing space)
if "Participant_ID" in df.columns:
    df.drop(columns=["Participant_ID"], inplace=True)

# Step 2: Define preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

categorical_cols = df.select_dtypes(include=["object"]).drop(columns=["CRC_Risk"], errors="ignore").columns.tolist()
numerical_cols = df.select_dtypes(include=["int64", "float64"]).drop(columns=["CRC_Risk"], errors="ignore").columns.tolist()

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# Step 3: Prepare data
from sklearn.model_selection import train_test_split

X = df.drop("CRC_Risk", axis=1)
y = df["CRC_Risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train model
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)

# Step 5: Predict probabilities
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]  # Probability of high-risk (class 1)

# Step 6: Use threshold to classify
threshold = 0.5
y_pred_class = (y_pred_proba >= threshold).astype(int)

from sklearn.metrics import classification_report, confusion_matrix

print("Classification Report (threshold = 0.5):")
print(classification_report(y_test, y_pred_class))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_class))

# Show top 5 predicted probabilities
print("Top 5 high-risk probabilities on test data:")
print(y_pred_proba[:5])

# Step 7: Predict for new unseen data
new_data = X.iloc[[0]].copy()
new_data["Age"] = 70
new_data["BMI"] = 29.5

new_proba = pipeline.predict_proba(new_data)[0][1]  # probability of class 1
print(f"\nNew Data High Risk Probability: {new_proba:.2f}")

# Flag based on custom threshold
custom_threshold = 0.4
flag = int(new_proba > custom_threshold)
print(f"Risk Flag (threshold = {custom_threshold}):", flag)

# Step 8: Save model
import joblib
joblib.dump(pipeline, "crc_risk_model.joblib")
print("Model saved as: crc_risk_model.joblib")
