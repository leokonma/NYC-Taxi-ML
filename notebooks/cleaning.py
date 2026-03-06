"""
taxi_pipeline.py

A small, well-structured ETL/feature-engineering pipeline for the
NYC Yellow Taxi 2019 master table.

Goal
----
Return a cleaned and enriched DataFrame ready for anomaly modeling.

Design Principles
-----------------
- Deterministic and reproducible (randomness is not used here).
- Minimal but defensible cleaning (removes only structural corruption & physical impossibilities).
- Feature engineering focuses on behavior (physics + economics + time + spatial context).
- Functions are composable and testable: each step is isolated and documented.

Usage
-----
from taxi_pipeline import build_clean_model_df
df_model = build_clean_model_df(master_parquet_path, taxi_zone_lookup_path)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Configuration
# -----------------------------

@dataclass(frozen=True)
class CleaningConfig:
    """
    Thresholds for minimal physical plausibility checks.

    These bounds are chosen to remove only obviously impossible trips,
    NOT rare-but-valid behavior (which anomaly detection should capture).
    """
    max_speed_mph: float = 120.0
    max_duration_min: float = 6 * 60  # 6 hours
    congestion_fee_threshold: float = 1.0  # separates near-zero cluster vs ~2.5 cluster


CORE_VENDOR_COLS: list[str] = [
    "VendorID",
    "RatecodeID",
    "store_and_fwd_flag",
    "payment_type",
    "passenger_count",
]


# -----------------------------
# Helpers
# -----------------------------

def find_project_root(start: Optional[Path] = None) -> Path:
    """
    Walk up parent directories until we find a folder containing 'data/'.

    This is useful if the script is called from different locations (notebooks/, src/, etc.).
    """
    start_path = Path().resolve() if start is None else start.resolve()
    for p in [start_path] + list(start_path.parents):
        if (p / "data").exists():
            return p
    raise FileNotFoundError("Could not find project root containing a 'data' directory.")


def load_master_table(master_parquet_path: Path) -> pd.DataFrame:
    """Load the master parquet table."""
    return pd.read_parquet(master_parquet_path)


def ensure_datetimes(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """
    Ensure datetime columns are parsed correctly.
    Coerces invalid strings to NaT.
    """
    df = df.copy()
    for c in cols:
        if c in df.columns and df[c].dtype == "object":
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


# -----------------------------
# Cleaning steps
# -----------------------------

def drop_structural_missing_vendor_block(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where a block of vendor-related columns are ALL missing.

    Motivation:
    - These rows are usually ingestion/reporting artifacts.
    - They are not meaningful behavioral anomalies; they are data corruption.
    """
    df = df.copy()
    mask = df[CORE_VENDOR_COLS].isna().all(axis=1)
    return df.loc[~mask].copy()


def drop_missing_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where pickup or dropoff datetime is missing."""
    return df.dropna(subset=["tpep_pickup_datetime", "tpep_dropoff_datetime"]).copy()


def drop_constant_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Drop columns if they exist (e.g. year if constant in 2019-only data)."""
    df = df.copy()
    cols_to_drop = [c for c in cols if c in df.columns]
    return df.drop(columns=cols_to_drop).copy()


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic physics features:
    - duration_min
    - speed_mph
    """
    df = df.copy()

    df["duration_min"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60.0

    df["speed_mph"] = df["trip_distance"] / (df["duration_min"] / 60.0)
    return df


def filter_physical_plausibility(df: pd.DataFrame, cfg: CleaningConfig) -> pd.DataFrame:
    """
    Remove trips that are physically impossible.

    We remove only:
    - non-positive duration
    - non-positive distance
    - non-positive total_amount
    - extreme speed
    - extreme duration
    """
    df = df.copy()

    base = (
        (df["duration_min"] > 0) &
        (df["trip_distance"] > 0) &
        (df["total_amount"] > 0)
    )
    df = df.loc[base].copy()

    plausible = (
        (df["speed_mph"] > 0) &
        (df["speed_mph"] < cfg.max_speed_mph) &
        (df["duration_min"] < cfg.max_duration_min)
    )
    return df.loc[plausible].copy()


def add_congestion_indicator(df: pd.DataFrame, cfg: CleaningConfig) -> pd.DataFrame:
    """
    Convert congestion surcharge into a binary indicator.

    Rationale:
    - The distribution is bimodal: ~0 vs ~2.5 (with rounding noise).
    - Treating it as continuous injects meaningless numeric granularity.
    """
    df = df.copy()
    if "congestion_surcharge" in df.columns:
        df["has_congestion_fee"] = (df["congestion_surcharge"] > cfg.congestion_fee_threshold).astype(int)
        df = df.drop(columns=["congestion_surcharge"])
    elif "has_congestion_fee" not in df.columns:
        # If the surcharge column isn't present, create a conservative default
        df["has_congestion_fee"] = 0
    return df


# -----------------------------
# Feature engineering
# -----------------------------

def add_economic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ratio-based economic features.

    Notes:
    - Use small eps in denominators to avoid division-by-zero infinities.
    - tip_pct: cash tips are often unobserved; we will also add tip_observed in add_tip_observed().
    """
    df = df.copy()
    eps = 1e-6

    df["fare_per_mile"] = df["fare_amount"] / (df["trip_distance"] + eps)
    df["total_per_mile"] = df["total_amount"] / (df["trip_distance"] + eps)

    # tip_pct: NaN if fare_amount is 0. We'll fill later after adding tip_observed.
    df["tip_pct"] = df["tip_amount"] / (df["fare_amount"].replace(0, np.nan))

    df["toll_ratio"] = df["tolls_amount"] / (df["total_amount"].replace(0, np.nan))
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time features and cyclical encodings for hour/month."""
    df = df.copy()

    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["weekday"] = df["tpep_pickup_datetime"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["pickup_hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["pickup_hour"] / 24.0)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    return df


def add_spatial_boroughs(df: pd.DataFrame, taxi_zone_lookup_path: Path) -> pd.DataFrame:
    """
    Merge borough information from taxi zone lookup table.

    This is crucial because PULocationID/DOLocationID are categorical IDs, NOT numeric magnitudes.
    Borough provides real geographic structure without injecting fake ordinal relationships.
    """
    df = df.copy()

    zones = pd.read_csv(taxi_zone_lookup_path)
    # Expect TLC schema: LocationID, Borough, Zone, service_zone
    keep = ["LocationID", "Borough"]
    zones = zones[keep].copy()

    # Pickup borough
    zones_pu = zones.rename(columns={"LocationID": "PULocationID", "Borough": "PU_Borough"})
    df = df.merge(zones_pu, on="PULocationID", how="left")

    # Dropoff borough
    zones_do = zones.rename(columns={"LocationID": "DOLocationID", "Borough": "DO_Borough"})
    df = df.merge(zones_do, on="DOLocationID", how="left")

    return df


def add_log_features(df: pd.DataFrame) -> pd.DataFrame:
    """Log-transform heavy-tailed positive variables."""
    df = df.copy()
    df["log_duration"] = np.log1p(df["duration_min"])
    df["log_distance"] = np.log1p(df["trip_distance"])
    return df


def add_expected_speed_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create contextual speed features:
    - expected_speed = median(speed_mph | pickup_hour, PU_Borough)
    - speed_deviation = speed_mph - expected_speed

    This enables contextual anomalies (speed unusual for a given hour+borough).
    """
    df = df.copy()

    # Requires PU_Borough and pickup_hour; if missing, skip
    if "PU_Borough" not in df.columns or "pickup_hour" not in df.columns:
        df["expected_speed"] = np.nan
        df["speed_deviation"] = np.nan
        return df

    baseline = df.groupby(["pickup_hour", "PU_Borough"])["speed_mph"].median()

    idx = df.set_index(["pickup_hour", "PU_Borough"]).index
    df["expected_speed"] = idx.map(baseline)
    df["speed_deviation"] = df["speed_mph"] - df["expected_speed"]

    return df


def add_tip_observed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a feature indicating whether tip is typically observed.

    TLC convention (commonly):
    payment_type:
      1 = Credit card (tip is recorded)
      2 = Cash (tip may be unobserved in the data)
      ... other codes exist

    We encode:
    tip_observed = 1 if payment_type == 1 else 0
    Then fill tip_pct NaNs with 0 (so missingness is represented separately).
    """
    df = df.copy()
    if "payment_type" in df.columns:
        df["tip_observed"] = (df["payment_type"] == 1).astype(int)
    else:
        df["tip_observed"] = 0

    # After encoding observability, fill NaNs safely
    if "tip_pct" in df.columns:
        df["tip_pct"] = df["tip_pct"].fillna(0)
    if "toll_ratio" in df.columns:
        df["toll_ratio"] = df["toll_ratio"].fillna(0)

    return df


# -----------------------------
# Public API
# -----------------------------

def build_clean_model_df(
    master_parquet_path: Path,
    taxi_zone_lookup_path: Path,
    cfg: Optional[CleaningConfig] = None,
) -> pd.DataFrame:
    """
    End-to-end pipeline:
    load -> clean -> enrich -> return DataFrame ready to model.

    Parameters
    ----------
    master_parquet_path:
        Path to the master 2019 parquet table.
    taxi_zone_lookup_path:
        Path to the taxi zone lookup CSV.
    cfg:
        CleaningConfig controlling plausibility thresholds.

    Returns
    -------
    pd.DataFrame
        Cleaned and feature-engineered dataset.
    """
    cfg = CleaningConfig() if cfg is None else cfg

    df = load_master_table(master_parquet_path)

    df = drop_structural_missing_vendor_block(df)
    df = ensure_datetimes(df, ["tpep_pickup_datetime", "tpep_dropoff_datetime"])
    df = drop_missing_datetimes(df)
    df = drop_constant_columns(df, ["year"])  # you requested dropping year

    df = add_physics_features(df)
    df = filter_physical_plausibility(df, cfg)

    df = add_congestion_indicator(df, cfg)

    # Enrichment
    df = add_economic_features(df)
    df = add_time_features(df)
    df = add_spatial_boroughs(df, taxi_zone_lookup_path)
    df = add_log_features(df)
    df = add_expected_speed_features(df)
    df = add_tip_observed(df)

    return df


# Optional: quick CLI-like smoke test
if __name__ == "__main__":
    root = find_project_root()
    master_path = root / "data" / "procesed" / "master_2019_1M_per_month.parquet"
    zones_path = root / "data" / "raw" / "taxi_zone_lookup.csv"

    df_model = build_clean_model_df(master_path, zones_path)
    print(df_model.shape)
    print(df_model.dtypes)