import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from features import engineer_features, get_feature_columns, load_data


# ~2.5% contamination — tuned to match the observed rate of obvious
# physics violations (negative power, voltage drops, zero current, etc.)
CONTAMINATION = 0.025
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def apply_hard_rules(df: pd.DataFrame) -> np.ndarray:
    # catch clear-cut violations that the model might score borderline
    rules = (
        (df["negative_power"] == 1)
        | (df["current_zero"] == 1)
        | (df["voltage_out_of_range"] == 1)
        | (df["overtemp"] == 1)
        | (df["has_error"] == 1)
    )
    return rules.values


def train(data_path: str):
    print(f"Loading data from {data_path}...")
    df = load_data(data_path)
    print(f"  {len(df):,} events, {df['station_id'].nunique()} stations, "
          f"{df['session_id'].nunique()} sessions")

    print("Engineering features...")
    df = engineer_features(df)

    feature_cols = get_feature_columns()
    X = df[feature_cols].copy()

    # fill NaNs that come from rolling windows at the start of each station's history
    X = X.fillna(X.median())

    print("Fitting scaler and Isolation Forest...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # -1 = anomaly, 1 = normal  →  convert to 0/1
    if_labels = (model.predict(X_scaled) == -1).astype(int)
    hard_rule_labels = apply_hard_rules(df).astype(int)

    # union: flag if either the model or a hard rule fires
    final_labels = np.clip(if_labels + hard_rule_labels, 0, 1)

    total = len(final_labels)
    flagged = final_labels.sum()
    print(f"\nResults on training data:")
    print(f"  Total events      : {total:,}")
    print(f"  Flagged anomalies : {flagged:,} ({flagged/total*100:.2f}%)")
    print(f"    - Isolation Forest only : {((if_labels == 1) & (hard_rule_labels == 0)).sum():,}")
    print(f"    - Hard rules only       : {((if_labels == 0) & (hard_rule_labels == 1)).sum():,}")
    print(f"    - Both                  : {((if_labels == 1) & (hard_rule_labels == 1)).sum():,}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "isolation_forest.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    print(f"\nModel saved to {MODELS_DIR}/")


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "charging_logs.csv"
    train(data_path)
