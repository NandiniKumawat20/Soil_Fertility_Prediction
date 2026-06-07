import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def main():
    data_path = "Fertility_Dataset.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    if "Fertilizer Name" not in df.columns:
        raise ValueError("Expected column 'Fertilizer Name' not found in Fertility_Dataset.csv")

    target_col = "NPK_Distance_Score"
    if target_col not in df.columns:
        raise ValueError(f"Expected column '{target_col}' not found in Fertility_Dataset.csv")

    drop_features = {
        "EC", "B", "S", "Cu",
    }

    feature_cols = [
        c for c in df.columns
        if c not in ["Output", "Fertilizer Name", target_col] and c not in drop_features
    ]

    X = df[feature_cols].copy()
    y = df[target_col].copy()


    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in feature_cols if c not in numeric_features]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )

    # "Simple" model: RandomForest is fairly robust and works well out-of-box
    model = RandomForestRegressor(
        n_estimators=250,
        random_state=42,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
    )

    reg = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"[Fertility_Dataset] MAE (NPK_Distance_Score): {mae:.4f}")

    # Save
    joblib.dump(reg, "fertility_dataset_npk_distance_rf.joblib")
    print("Saved: fertility_dataset_npk_distance_rf.joblib")


if __name__ == "__main__":
    main()

