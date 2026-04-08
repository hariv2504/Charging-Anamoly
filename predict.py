import argparse
import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from features import engineer_features, get_feature_columns, load_data

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def apply_hard_rules(df: pd.DataFrame) -> np.ndarray:
    rules = (
        (df["negative_power"] == 1)
        | (df["current_zero"] == 1)
        | (df["voltage_out_of_range"] == 1)
        | (df["overtemp"] == 1)
        | (df["has_error"] == 1)
    )
    return rules.values


def predict(input_path: str, output_path: str):
    model_path = os.path.join(MODELS_DIR, "isolation_forest.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Model not found. Run `python src/train.py <data.csv>` first.")
        sys.exit(1)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    print(f"Loading {input_path}...")
    df = load_data(input_path)
    df = engineer_features(df)

    feature_cols = get_feature_columns()
    X = df[feature_cols].copy().fillna(df[feature_cols].median())

    X_scaled = scaler.transform(X)
    if_labels = (model.predict(X_scaled) == -1).astype(int)
    hard_rule_labels = apply_hard_rules(df).astype(int)

    df["is_anomaly"] = np.clip(if_labels + hard_rule_labels, 0, 1)

    # only keep original columns + is_anomaly in the output
    original_cols = [
        "station_id", "timestamp", "session_id", "voltage", "current",
        "power_kw", "temperature_c", "duration_sec", "energy_kwh",
        "error_code", "message", "is_anomaly",
    ]
    df[original_cols].to_csv(output_path, index=False)

    flagged = df["is_anomaly"].sum()
    print(f"Done. {flagged:,} anomalies flagged out of {len(df):,} events "
          f"({flagged/len(df)*100:.2f}%)")
    print(f"Output written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect anomalies in EV charging logs")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to output CSV")
    args = parser.parse_args()
    predict(args.input, args.output)