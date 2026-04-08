import numpy as np
import pandas as pd


# Voltage range for a healthy Level 2 charger (roughly 208–240V AC)
VOLTAGE_MIN = 190.0
VOLTAGE_MAX = 260.0
TEMP_MAX = 75.0  # degrees C — above this is a thermal concern


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- physics sanity flags ---
    # power should be positive during a charging event
    df["negative_power"] = (df["power_kw"] < 0).astype(int)

    # current=0 with non-zero power is physically inconsistent
    df["current_zero"] = (df["current"] == 0).astype(int)

    # voltage outside expected operating range
    df["voltage_out_of_range"] = (
        (df["voltage"] < VOLTAGE_MIN) | (df["voltage"] > VOLTAGE_MAX)
    ).astype(int)

    # thermal flag
    df["overtemp"] = (df["temperature_c"] > TEMP_MAX).astype(int)

    # error code present
    df["has_error"] = (df["error_code"] != 0).astype(int)

    # --- power factor proxy ---
    # for AC: P = V * I * pf, so pf ≈ P / (V * I)
    # deviations from ~0.95–1.0 suggest metering issues
    apparent_power = df["voltage"] * df["current"] / 1000.0  # kVA
    df["power_factor"] = np.where(
        apparent_power > 0.1,
        df["power_kw"] / apparent_power,
        np.nan,
    )

    # --- energy consistency check ---
    # energy_kwh should roughly equal power_kw * duration_sec / 3600
    expected_energy = df["power_kw"] * df["duration_sec"] / 3600.0
    df["energy_ratio"] = np.where(
        expected_energy.abs() > 0.001,
        df["energy_kwh"] / expected_energy.abs(),
        np.nan,
    )

    # --- station-level rolling stats (window=20 events per station) ---
    # gives a local baseline so we can detect drift within a session
    for col in ["voltage", "power_kw", "temperature_c"]:
        rolled = (
            df.groupby("station_id")[col]
            .transform(lambda x: x.rolling(20, min_periods=5).mean())
        )
        df[f"{col}_roll_mean"] = rolled

        rolled_std = (
            df.groupby("station_id")[col]
            .transform(lambda x: x.rolling(20, min_periods=5).std())
        )
        df[f"{col}_roll_std"] = rolled_std

        # z-score relative to rolling window
        df[f"{col}_zscore"] = np.where(
            rolled_std > 0,
            (df[col] - rolled) / rolled_std,
            0.0,
        )

    # --- session-level aggregates ---
    session_stats = df.groupby("session_id").agg(
        session_mean_power=("power_kw", "mean"),
        session_std_power=("power_kw", "std"),
        session_max_temp=("temperature_c", "max"),
        session_error_count=("has_error", "sum"),
        session_event_count=("power_kw", "count"),
    ).reset_index()
    session_stats["session_std_power"] = session_stats["session_std_power"].fillna(0)
    df = df.merge(session_stats, on="session_id", how="left")

    # --- time features ---
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek

    return df


def get_feature_columns() -> list:
    # the columns we actually feed into the model
    return [
        "voltage",
        "current",
        "power_kw",
        "temperature_c",
        "duration_sec",
        "energy_kwh",
        "negative_power",
        "current_zero",
        "voltage_out_of_range",
        "overtemp",
        "has_error",
        "power_factor",
        "energy_ratio",
        "voltage_zscore",
        "power_kw_zscore",
        "temperature_c_zscore",
        "session_mean_power",
        "session_std_power",
        "session_max_temp",
        "session_error_count",
        "hour",
        "dayofweek",
    ]
