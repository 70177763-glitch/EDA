"""
filters.py – Data loading, cleaning, and filtering functions
Course   : Exploratory Data Analysis
Instructor: Ali Hassan Sherazi
Dataset  : Estimates_2025_en.csv

WHY: Separating filter logic keeps app.py clean and makes each function
individually testable and reusable.
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ── 1. LOAD & CLEAN ──────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """
    Load and clean the dataset from disk.
    """

    # Get project folder path
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "data" / "Estimates_2025_en.csv"

    # Load CSV
    df = pd.read_csv(csv_path, low_memory=False)

    # Drop unnecessary columns
    df.drop(
        columns=[
            "Indicator_GId",
            "Subgroup_Val_GId",
            "Data_Denominator",
            "Footnote",
        ],
        errors="ignore",
        inplace=True,
    )

    # Rename columns
    df.rename(
        columns={
            "Data value": "Data_Value",
            "Area ID": "Area_ID",
            "Area Level": "Area_Level",
            "Time Period": "Time_Period",
        },
        inplace=True,
    )

    # Convert numeric columns
    if "Data_Value" in df.columns:
        df["Data_Value"] = pd.to_numeric(df["Data_Value"], errors="coerce")

    if "Time_Period" in df.columns:
        df["Time_Period"] = pd.to_numeric(df["Time_Period"], errors="coerce")

    if "Area_Level" in df.columns:
        df["Area_Level"] = pd.to_numeric(df["Area_Level"], errors="coerce")

    # Remove missing values
    df.dropna(subset=["Data_Value", "Time_Period"], inplace=True)

    # Remove negative values
    df = df[df["Data_Value"] >= 0].copy()

    # Clean text columns
    for col in ["Indicator", "Unit", "Subgroup", "Area", "Source"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().fillna("Unknown")

    # Reset index
    df.reset_index(drop=True, inplace=True)

    return df


# ── 2. FILTER FUNCTION ───────────────────────────────────────────────────────

def apply_filters(
    df: pd.DataFrame,
    areas: list,
    indicators: list,
    units: list,
    sources: list,
    area_levels: list,
    subgroups: list,
    time_range: tuple,
    value_range: tuple,
) -> pd.DataFrame:

    fdf = df.copy()

    # Apply filters safely
    if areas:
        fdf = fdf[fdf["Area"].isin(areas)]

    if indicators:
        fdf = fdf[fdf["Indicator"].isin(indicators)]

    if units:
        fdf = fdf[fdf["Unit"].isin(units)]

    if sources:
        fdf = fdf[fdf["Source"].isin(sources)]

    if area_levels and "Area_Level" in fdf.columns:
        fdf = fdf[fdf["Area_Level"].isin(area_levels)]

    if subgroups:
        fdf = fdf[fdf["Subgroup"].isin(subgroups)]

    # Time range filter
    fdf = fdf[
        (fdf["Time_Period"] >= time_range[0]) &
        (fdf["Time_Period"] <= time_range[1])
    ]

    # Value range filter
    fdf = fdf[
        (fdf["Data_Value"] >= value_range[0]) &
        (fdf["Data_Value"] <= value_range[1])
    ]

    return fdf.reset_index(drop=True)