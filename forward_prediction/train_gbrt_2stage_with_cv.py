from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid
from sklearn.multioutput import MultiOutputRegressor


# -----------------------------
# 1) Load data
# -----------------------------
DATA_DIR = Path("forward_prediction/split")
RESULTS_FILE = Path("results.txt")

X_train = pd.read_csv(DATA_DIR / "train_features.csv")
X_test = pd.read_csv(DATA_DIR / "test_features.csv")
y_train = pd.read_csv(DATA_DIR / "train_labels.csv")
y_test = pd.read_csv(DATA_DIR / "test_labels.csv")

# Remove row_id if present (it is an index, not a real feature/target).
for df in (X_train, X_test, y_train, y_test):
    if "row_id" in df.columns:
        df.drop(columns=["row_id"], inplace=True)

print("Data loaded.")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test : {X_test.shape}, y_test : {y_test.shape}")


# -----------------------------
# 2) Definitions
# -----------------------------
STAGE1_FEATURES = [
    "porosity",
    "atmosphere",
    "coupling",
    "strength",
    "shape_factor",
    "energy",
    "angle_rad",
]  # gravity intentionally excluded

STAGE1_TARGETS = ["P80", "fines_frac", "oversize_frac"]
STAGE2_TARGETS = ["R95", "R50_fines", "R50_oversize"]

required_columns = set(STAGE1_FEATURES + ["gravity"])
missing = required_columns - set(X_train.columns)
if missing:
    raise ValueError(f"Missing required feature columns: {sorted(missing)}")

missing_targets = set(STAGE1_TARGETS + STAGE2_TARGETS) - set(y_train.columns)
if missing_targets:
    raise ValueError(f"Missing required target columns: {sorted(missing_targets)}")

param_grid = {
    "estimator__n_estimators": [180, 200, 220], #number of boosting stages (number of trees)
    "estimator__learning_rate": [0.015, 0.025, 0.035], #step size of each boosting stage
    "estimator__max_depth": [4, 5, 6], #maximum depth of each regression tree 
    "estimator__subsample": [0.5, 0.6], #fraction of samples used for fitting each base learner
    "estimator__min_samples_leaf": [5, 10, 12], #minimum number of samples required to be at a leaf node (= minimum number of samples in each child node to do a split)
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)


def run_grid_search(stage_name: str, X: pd.DataFrame, y: pd.DataFrame) -> GridSearchCV:
    num_candidates = len(ParameterGrid(param_grid))
    total_fits = num_candidates * cv.get_n_splits()

    print(f"\n{stage_name}: starting hyperparameter search...")
    print(f"Candidates: {num_candidates}")
    print(f"CV folds  : {cv.get_n_splits()}")
    print(f"Total fits: {total_fits}")

    model = MultiOutputRegressor(GradientBoostingRegressor(random_state=42))

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    start = time.time()
    search.fit(X, y)
    elapsed = time.time() - start

    print(f"\n{stage_name}: best hyperparameters")
    print(search.best_params_)
    print(f"{stage_name}: best CV MAE = {-search.best_score_:.6f}")
    print(f"{stage_name}: training time = {elapsed:.1f} seconds")
    return search


# -----------------------------
# 3) Stage 1 model
# -----------------------------
X1_train = X_train[STAGE1_FEATURES]
X1_test = X_test[STAGE1_FEATURES]
y1_train = y_train[STAGE1_TARGETS]

stage1_search = run_grid_search("Stage 1 (Fragment model)", X1_train, y1_train)
stage1_best_model = stage1_search.best_estimator_

# Build out-of-fold predictions for train set (used as Stage 2 inputs)
print("\nStage 1: generating out-of-fold predictions for Stage 2 training...")
stage1_best_params = {
    key.replace("estimator__", ""): value
    for key, value in stage1_search.best_params_.items()
}

stage1_oof_pred = np.zeros((len(X1_train), len(STAGE1_TARGETS)))
for fold_idx, (idx_tr, idx_val) in enumerate(cv.split(X1_train), start=1):
    fold_model = MultiOutputRegressor(
        GradientBoostingRegressor(random_state=42, **stage1_best_params)
    )
    fold_model.fit(X1_train.iloc[idx_tr], y1_train.iloc[idx_tr])
    stage1_oof_pred[idx_val, :] = fold_model.predict(X1_train.iloc[idx_val])
    print(f"Stage 1 OOF progress: fold {fold_idx}/{cv.get_n_splits()} done")

# Stage 1 predictions for test set
stage1_test_pred = stage1_best_model.predict(X1_test)


# -----------------------------
# 4) Stage 2 model
# -----------------------------
# Stage 2 uses ALL original features (including gravity) + Stage 1 outputs.
X2_train = X_train.copy()
X2_test = X_test.copy()

for i, target_name in enumerate(STAGE1_TARGETS):
    X2_train[f"pred_{target_name}"] = stage1_oof_pred[:, i]
    X2_test[f"pred_{target_name}"] = stage1_test_pred[:, i]

y2_train = y_train[STAGE2_TARGETS]

stage2_search = run_grid_search("Stage 2 (Distance model)", X2_train, y2_train)
stage2_best_model = stage2_search.best_estimator_
stage2_test_pred = stage2_best_model.predict(X2_test)


# -----------------------------
# 5) Join predictions + evaluate
# -----------------------------
y_pred_df = pd.DataFrame(index=y_test.index)
for i, target_name in enumerate(STAGE1_TARGETS):
    y_pred_df[target_name] = stage1_test_pred[:, i]
for i, target_name in enumerate(STAGE2_TARGETS):
    y_pred_df[target_name] = stage2_test_pred[:, i]

y_pred_df = y_pred_df[y_test.columns]
y_true = y_test.to_numpy()
y_pred = y_pred_df.to_numpy()

mae_overall = mean_absolute_error(y_true, y_pred)
rmse_overall = np.sqrt(mean_squared_error(y_true, y_pred))
r2_overall = r2_score(y_true, y_pred, multioutput="uniform_average")

print("\nTest set (overall):")
print(f"MAE  : {mae_overall:.6f}")
print(f"RMSE : {rmse_overall:.6f}")
print(f"R2   : {r2_overall:.6f}")


# -----------------------------
# 6) Challenge benchmark score
# -----------------------------
TARGET_WEIGHTS = {
    "P80": 0.30,
    "R95": 0.20,
    "fines_frac": 0.15,
    "oversize_frac": 0.15,
    "R50_fines": 0.10,
    "R50_oversize": 0.10,
}


# sMAPE (symmetric mean absolute percentage error) for fractional targets.
def smape(y_true_1d: np.ndarray, y_pred_1d: np.ndarray, eps: float = 1e-12) -> float:
    denominator = np.abs(y_true_1d) + np.abs(y_pred_1d) + eps
    return float(np.mean(2.0 * np.abs(y_pred_1d - y_true_1d) / denominator))


challenge_errors: dict[str, float] = {}
for target in TARGET_WEIGHTS:
    y_t = y_test[target].to_numpy()
    y_p = y_pred_df[target].to_numpy()

    if target in {"fines_frac", "oversize_frac"}:
        challenge_errors[target] = smape(y_t, y_p)
    else:
        challenge_errors[target] = float(mean_absolute_error(y_t, y_p))

weighted_error = sum(TARGET_WEIGHTS[t] * challenge_errors[t] for t in TARGET_WEIGHTS)
challenge_score = 100.0 / (1.0 + weighted_error)

print("\nChallenge benchmark:")
for target in TARGET_WEIGHTS:
    metric_name = "sMAPE" if target in {"fines_frac", "oversize_frac"} else "MAE"
    print(
        f"{target:>15} | {metric_name}={challenge_errors[target]:.6f}  "
        f"weight={TARGET_WEIGHTS[target]:.2f}"
    )
print(f"Weighted error: {weighted_error:.6f}")
print(f"Challenge score: {challenge_score:.6f}")

print("\nTest set (per target):")
for target_name in y_test.columns:
    y_t = y_test[target_name].to_numpy()
    y_p = y_pred_df[target_name].to_numpy()
    mae = mean_absolute_error(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    r2 = r2_score(y_t, y_p)
    print(f"{target_name:>15} | MAE={mae:.6f}  RMSE={rmse:.6f}  R2={r2:.6f}")


# -----------------------------
# 7) Append run summary to results.txt
# -----------------------------
result_lines = [
    "",
    "=" * 80,
    f"Script: {Path(__file__).name}",
    "Model specification:",
    "  - Two-stage MultiOutputRegressor(GradientBoostingRegressor(random_state=42))",
    f"  - Stage 1 features: {STAGE1_FEATURES}",
    f"  - Stage 1 targets: {STAGE1_TARGETS}",
    f"  - Stage 2 targets: {STAGE2_TARGETS}",
    f"  - Stage 1 best hyperparameters: {stage1_search.best_params_}",
    f"  - Stage 2 best hyperparameters: {stage2_search.best_params_}",
    "  - CV: KFold(n_splits=5, shuffle=True, random_state=42)",
    "  - Scoring: neg_mean_absolute_error",
    "",
    "Overall test metrics:",
    f"  MAE={mae_overall:.6f}",
    f"  RMSE={rmse_overall:.6f}",
    f"  R2={r2_overall:.6f}",
    "",
    "Challenge benchmark:",
]

for target in TARGET_WEIGHTS:
    metric_name = "sMAPE" if target in {"fines_frac", "oversize_frac"} else "MAE"
    result_lines.append(
        f"  {target}: {metric_name}={challenge_errors[target]:.6f}, weight={TARGET_WEIGHTS[target]:.2f}"
    )

result_lines.extend(
    [
        f"  Weighted error={weighted_error:.6f}",
        f"  Challenge score={challenge_score:.6f}",
        "",
        "Per-target test metrics:",
    ]
)

for target_name in y_test.columns:
    y_t = y_test[target_name].to_numpy()
    y_p = y_pred_df[target_name].to_numpy()
    mae = mean_absolute_error(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    r2 = r2_score(y_t, y_p)
    result_lines.append(
        f"  {target_name}: MAE={mae:.6f}, RMSE={rmse:.6f}, R2={r2:.6f}"
    )

with RESULTS_FILE.open("a", encoding="utf-8") as f:
    f.write("\n".join(result_lines) + "\n")

print(f"\nRun summary appended to {RESULTS_FILE}")
