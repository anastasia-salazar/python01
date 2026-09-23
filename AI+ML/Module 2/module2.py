from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

csv_path = Path(__file__).resolve().parent / "titanic.csv"


# ---------------------------------------------------------------
# 1. Data loading & selection
# ---------------------------------------------------------------
titanic = pd.read_csv(csv_path)
titanic.columns = titanic.columns.str.lower()  # "Survived" -> "survived", etc.

columns = ["survived", "pclass", "sex", "age", "fare", "embarked"]
df = titanic[columns].copy()
print(df.head())

print("\nMissing values per column:")
print(df.isna().sum())


# ---------------------------------------------------------------
# 2. Handling missing values
# ---------------------------------------------------------------
df["age"] = df["age"].fillna(df["age"].mean())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

print("\nMissing values after imputation:")
print(df.isna().sum())


# ---------------------------------------------------------------
# 3. Categorical encoding
# ---------------------------------------------------------------
# Label Encoding: female -> 0, male -> 1
label_encoder = LabelEncoder()
df["sex"] = label_encoder.fit_transform(df["sex"])
print("\nSex encoding:", dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))))

# One-Hot Encoding: embarked (C/Q/S) -> embarked_C, embarked_Q, embarked_S
df = pd.get_dummies(df, columns=["embarked"], dtype=int)


# ---------------------------------------------------------------
# 4. Feature scaling
# ---------------------------------------------------------------
scaler = StandardScaler()
df[["age", "fare"]] = scaler.fit_transform(df[["age", "fare"]])

print("\nAge/fare after scaling (mean ~0, std ~1):")
print(df[["age", "fare"]].agg(["mean", "std"]).round(3))

print("\nProcessed dataset:")
print(df.head())


# ---------------------------------------------------------------
# 5. Train-test split
# ---------------------------------------------------------------
X = df.drop(columns=["survived"])
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTraining set: {X_train.shape}, Test set: {X_test.shape}")


# ---------------------------------------------------------------
# 6. Preprocessing + model pipeline with cross-validation
# ---------------------------------------------------------------
# Steps 2-4 above were fitted on the whole dataset before splitting, so a
# little information from the test rows leaks into the training data.
# A Pipeline fixes that: it re-learns the mean/mode/scaling from the
# training rows only, inside every cross-validation fold.
raw = titanic[columns]
X_raw = raw.drop(columns=["survived"])
y_raw = raw["survived"]
X_raw_train, X_raw_test, y_raw_train, y_raw_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42
)

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler()),
    ]), ["age", "fare"]),
    ("embarked", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder()),
    ]), ["embarked"]),
    # A single binary column: one-hot with drop="if_binary" gives 0/1 like LabelEncoder
    ("sex", OneHotEncoder(drop="if_binary"), ["sex"]),
    ("pclass", "passthrough", ["pclass"]),
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

print("\n5-fold cross-validation accuracy (training set):")
results = {}
for name, model in models.items():
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    scores = cross_val_score(pipeline, X_raw_train, y_raw_train, cv=5, scoring="accuracy")
    results[name] = pipeline
    print(f"  {name:20s} {scores.mean():.4f} (+/- {scores.std():.4f})")

print("\nAccuracy on held-out test set:")
for name, pipeline in results.items():
    pipeline.fit(X_raw_train, y_raw_train)
    print(f"  {name:20s} {pipeline.score(X_raw_test, y_raw_test):.4f}")
