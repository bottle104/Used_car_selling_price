import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score as cv_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score

import joblib


# ==============================
# 1. LOAD DATA
# ==============================
def load_df(path):
    """Load dataset from CSV"""
    return pd.read_csv(path)


# ==============================
# 2. PREPROCESSING PIPELINES
# ==============================
def create_pipelines(cv=5):
    """Create linear and ridge pipelines"""

    pipe_linear = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    pipe_ridge = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=cv))
    ])

    return pipe_linear, pipe_ridge


# ==============================
# 3. TREE MODELS
# ==============================
def create_tree_models(n_estimators, max_depth, min_samples_leaf,
                        min_samples_split, max_features, n_jobs, random_state):
    """Create RandomForest and DecisionTree models"""

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        max_features=max_features,
        n_jobs=n_jobs,
        random_state=random_state
    )

    dt = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        random_state=random_state
    )

    return rf, dt


# ==============================
# 4. FEATURE ENGINEERING
# ==============================
def feature_engineering(df):
    """Extract numeric values from string columns"""

    for col in ["mileage", "engine", "max_power"]:
        df[col] = df[col].str.extract(r"(\d+\.?\d*)")[0].astype(float)

    return df


# ==============================
# 5. ONE HOT ENCODING
# ==============================
def one_hot_encode(df, columns, drop_first=False):
    """Convert categorical columns to numeric"""
    return pd.get_dummies(df, columns=columns, drop_first=drop_first)


# ==============================
# 6. SPLIT X AND Y
# ==============================
def split_features_target(df, target, drop_cols=None):
    """Separate features and target"""

    if drop_cols is None:
        drop_cols = []

    X = df.drop(columns=[target] + drop_cols)
    y = df[target]

    return X, y


# ==============================
# 7. TRAIN TEST SPLIT
# ==============================
def split_data(X, y, train_size=0.8, random_state=42):
    """Split dataset into train and test"""
    return train_test_split(X, y, train_size=train_size, random_state=random_state)


# ==============================
# 8. TRAIN MODELS
# ==============================
def train_models(models, x_train, y_train):
    """Train all models"""
    trained = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        trained[name] = model

    return trained


# ==============================
# 9. STORE RESULTS
# ==============================
def store_results(name, train_r2, test_r2, cv_r2, rmse, mae, results):
    """Store evaluation metrics"""
    results.append({
        "Model": name,
        "Train R2": train_r2,
        "Test R2": test_r2,
        "CV R2": cv_r2,
        "RMSE": rmse,
        "MAE": mae,
        "Overfit Gap": train_r2 - test_r2
    })


# ==============================
# 10. CREATE RESULT TABLE
# ==============================
def create_results_df(results):
    df = pd.DataFrame(results)
    return df.sort_values("Test R2", ascending=False)


# ==============================
# 11. DISPLAY RESULTS
# ==============================
def display_results(df):
    print("\n" + "=" * 80)
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)


# ==============================
# 12. FIND BEST MODEL
# ==============================
def find_best_model(df):
    best = df.iloc[0]
    print(f"\n🏆 BEST MODEL: {best['Model']}")
    print(f"   Test R2: {best['Test R2']:.4f}")
    print(f"   RMSE: {best['RMSE']:.4f}")
    return best["Model"]


# ==============================
# 13. OVERFITTING CHECK
# ==============================
def check_overfitting(df):
    overfit = df[df["Overfit Gap"] > 0.15]

    if not overfit.empty:
        print("\n⚠️ Overfitting detected:")
        for _, row in overfit.iterrows():
            print(f" - {row['Model']} (gap: {row['Overfit Gap']:.3f})")


# ==============================
# MAIN WORKFLOW
# ==============================

# Load data
df = load_df("Car details v3.csv")

# Preprocess
df = feature_engineering(df)
df = one_hot_encode(df, ["fuel", "seller_type", "transmission", "owner"])

# Split features
X, y = split_features_target(df, "selling_price", ["torque", "name"])
x_train, x_test, y_train, y_test = split_data(X, y)

# Create models
pipe1, pipe2 = create_pipelines()
rf, dt = create_tree_models(100, None, 1, 2, "sqrt", -1, 42)

models = {
    "Linear": pipe1,
    "Ridge": pipe2,
    "RandomForest": rf,
    "DecisionTree": dt
}

# Train
trained_models = train_models(models, x_train, y_train)

# Evaluate
results = []

for name, model in trained_models.items():

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    cv_r2 = cv_score(model, X, y, scoring="r2", cv=5).mean()

    rmse = root_mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)

    store_results(name, train_r2, test_r2, cv_r2, rmse, mae, results)

# Results
results_df = create_results_df(results)
display_results(results_df)
best_model = find_best_model(results_df)
check_overfitting(results_df)


# ==============================
# 14. TUNE RANDOM FOREST
# ==============================

rf_tuned = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42
)

rf_tuned.fit(x_train, y_train)

y_pred = rf_tuned.predict(x_test)

print("\nTuned RandomForest:")
print("R2:", r2_score(y_test, y_pred))
print("RMSE:", root_mean_squared_error(y_test, y_pred))

# Save model
joblib.dump(rf_tuned, "RandomForest_tuned.pkl")