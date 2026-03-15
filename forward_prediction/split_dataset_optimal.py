from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

"""
“optimal” deterministic split by:
using target-aware stratification (multi-target rank composite),
searching many candidate random splits (--trials) 
and selecting the one with the lowest distribution gap across both X and y.
"""


EPS = 1e-12


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic, distribution-balanced train/test split for "
            "forward_prediction/train.csv and forward_prediction/train_labels.csv."
        )
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("forward_prediction/train.csv"),
        help="Path to train features CSV.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("forward_prediction/train_labels.csv"),
        help="Path to train labels CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("forward_prediction/split"),
        help="Directory where split CSV files and metadata will be saved.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test ratio in (0,1). Default: 0.2",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1500,
        help="How many random candidate splits to evaluate. More = better but slower.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master seed for reproducible search.",
    )
    parser.add_argument(
        "--max-strata",
        type=int,
        default=20,
        help="Maximum quantile strata used for target-aware stratification.",
    )
    return parser


def _validate_inputs(x_path: Path, y_path: Path, test_size: float, trials: int) -> None:
    if not x_path.exists():
        raise FileNotFoundError(f"Features file not found: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Labels file not found: {y_path}")
    if not (0.0 < test_size < 1.0):
        raise ValueError("--test-size must be between 0 and 1 (exclusive).")
    if trials < 1:
        raise ValueError("--trials must be >= 1")


def _build_strata(y: pd.DataFrame, max_strata: int) -> pd.Series:
    """
    Build robust strata for multi-target regression:
    1) Rank-normalize each target to [0,1]
    2) Average those ranks into one composite target signal
    3) Quantile-bin that signal with as many strata as feasible
    """
    rank_df = y.rank(method="average", pct=True)
    composite = rank_df.mean(axis=1)

    for n_bins in range(max_strata, 1, -1):
        strata = pd.qcut(composite, q=n_bins, labels=False, duplicates="drop")
        counts = strata.value_counts(dropna=False)
        if strata.nunique(dropna=True) >= 2 and counts.min() >= 2:
            return strata.astype(int)

    # fallback: simple 2-bin median split (always possible for non-constant target)
    median = composite.median()
    strata = (composite >= median).astype(int)
    if strata.nunique() < 2:
        # degenerate edge case: all targets constant
        strata = pd.Series(np.zeros(len(y), dtype=int), index=y.index)
    return strata


def _stratified_indices(
    strata: pd.Series,
    test_size: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    test_idx_parts: list[np.ndarray] = []

    for label in np.sort(strata.unique()):
        idx = np.flatnonzero(strata.to_numpy() == label)
        group_size = idx.size

        # keep both train and test represented when possible
        n_test = int(round(group_size * test_size))
        if group_size >= 2:
            n_test = max(1, min(group_size - 1, n_test))
        else:
            n_test = 0

        chosen = rng.choice(idx, size=n_test, replace=False) if n_test > 0 else np.array([], dtype=int)
        test_idx_parts.append(chosen)

    test_idx = np.concatenate(test_idx_parts) if test_idx_parts else np.array([], dtype=int)
    test_idx.sort()

    train_mask = np.ones(len(strata), dtype=bool)
    train_mask[test_idx] = False
    train_idx = np.flatnonzero(train_mask)
    return train_idx, test_idx


def _column_distribution_gap(train: pd.Series, test: pd.Series, full: pd.Series) -> float:
    full_std = float(full.std(ddof=0)) + EPS
    iqr = float(full.quantile(0.75) - full.quantile(0.25)) + EPS

    mean_gap = abs(float(train.mean()) - float(test.mean())) / full_std
    std_gap = abs(float(train.std(ddof=0)) - float(test.std(ddof=0))) / full_std

    q_levels = [0.1, 0.25, 0.5, 0.75, 0.9]
    q_train = train.quantile(q_levels).to_numpy(dtype=float)
    q_test = test.quantile(q_levels).to_numpy(dtype=float)
    quantile_gap = float(np.mean(np.abs(q_train - q_test) / iqr))

    return mean_gap + 0.7 * std_gap + 0.7 * quantile_gap


def _split_score(
    x: pd.DataFrame,
    y: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    target_test_size: float,
) -> float:
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    x_gap = np.mean([
        _column_distribution_gap(x_train[col], x_test[col], x[col])
        for col in x.columns
    ])
    y_gap = np.mean([
        _column_distribution_gap(y_train[col], y_test[col], y[col])
        for col in y.columns
    ])

    actual_test_ratio = len(test_idx) / len(x)
    size_penalty = abs(actual_test_ratio - target_test_size)

    # prioritize labels, then features, then exact ratio
    return 0.6 * y_gap + 0.35 * x_gap + 0.05 * size_penalty


def find_best_split(
    x: pd.DataFrame,
    y: pd.DataFrame,
    test_size: float,
    trials: int,
    seed: int,
    max_strata: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    strata = _build_strata(y, max_strata=max_strata)
    rng_master = np.random.default_rng(seed)

    best_score = float("inf")
    best_train_idx: np.ndarray | None = None
    best_test_idx: np.ndarray | None = None
    best_seed = None

    for _ in range(trials):
        candidate_seed = int(rng_master.integers(0, 2**31 - 1))
        rng = np.random.default_rng(candidate_seed)

        train_idx, test_idx = _stratified_indices(strata=strata, test_size=test_size, rng=rng)

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        score = _split_score(x, y, train_idx, test_idx, target_test_size=test_size)
        if score < best_score:
            best_score = score
            best_train_idx = train_idx
            best_test_idx = test_idx
            best_seed = candidate_seed

    if best_train_idx is None or best_test_idx is None or best_seed is None:
        raise RuntimeError("Could not find a valid split. Check dataset variability and parameters.")

    metadata = {
        "master_seed": seed,
        "best_candidate_seed": best_seed,
        "trials": trials,
        "test_size_requested": test_size,
        "test_size_actual": len(best_test_idx) / len(x),
        "best_score": best_score,
        "n_rows_total": len(x),
        "n_rows_train": int(len(best_train_idx)),
        "n_rows_test": int(len(best_test_idx)),
        "n_features": int(x.shape[1]),
        "n_targets": int(y.shape[1]),
        "n_strata_used": int(_build_strata(y, max_strata=max_strata).nunique()),
    }
    return best_train_idx, best_test_idx, metadata


def run(
    features_path: Path,
    labels_path: Path,
    output_dir: Path,
    test_size: float,
    trials: int,
    seed: int,
    max_strata: int,
) -> None:
    _validate_inputs(features_path, labels_path, test_size, trials)

    x = pd.read_csv(features_path)
    y = pd.read_csv(labels_path)

    if len(x) != len(y):
        raise ValueError(
            f"Row count mismatch: features has {len(x)} rows, labels has {len(y)} rows."
        )

    # keep original row id so split can always be traced back exactly
    x = x.copy()
    y = y.copy()
    x.insert(0, "row_id", np.arange(len(x), dtype=int))
    y.insert(0, "row_id", np.arange(len(y), dtype=int))

    train_idx, test_idx, metadata = find_best_split(
        x=x.drop(columns=["row_id"]),
        y=y.drop(columns=["row_id"]),
        test_size=test_size,
        trials=trials,
        seed=seed,
        max_strata=max_strata,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    x_train = x.iloc[train_idx].reset_index(drop=True)
    x_test = x.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    x_train.to_csv(output_dir / "train_features.csv", index=False)
    x_test.to_csv(output_dir / "test_features.csv", index=False)
    y_train.to_csv(output_dir / "train_labels.csv", index=False)
    y_test.to_csv(output_dir / "test_labels.csv", index=False)

    with (output_dir / "split_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Split complete.")
    print(f"Rows: total={metadata['n_rows_total']}, train={metadata['n_rows_train']}, test={metadata['n_rows_test']}")
    print(f"Best candidate seed: {metadata['best_candidate_seed']}")
    print(f"Best score: {metadata['best_score']:.6f}")
    print(f"Saved files to: {output_dir.resolve()}")


def main() -> None:
    args = build_parser().parse_args()
    run(
        features_path=args.features,
        labels_path=args.labels,
        output_dir=args.output_dir,
        test_size=args.test_size,
        trials=args.trials,
        seed=args.seed,
        max_strata=args.max_strata,
    )


if __name__ == "__main__":
    main()
