"""
Data normalization layer for the Revenue Leakage Detection system.

Converts messy CSV/Excel values into consistent values that can
be processed by the ML pipeline.
"""

import re
import numpy as np
import pandas as pd

from .schema import (
    NUMERICAL_FIELDS,
    DATE_FIELDS,
)


# ============================================================
# COMMON MISSING VALUES
# ============================================================

MISSING_VALUES = {
    "",
    " ",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "-",
    "--",
    "unknown",
    "not available",
    "not_applicable",
}


# ============================================================
# NORMALIZE STRING
# ============================================================

def normalize_string(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value.lower() in MISSING_VALUES:
        return np.nan

    # Remove unnecessary spaces
    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value


# ============================================================
# NORMALIZE NUMERIC VALUE
# ============================================================

def normalize_numeric(value):

    if pd.isna(value):
        return np.nan

    # Already numeric
    if isinstance(value, (int, float, np.integer, np.floating)):

        if np.isfinite(value):
            return float(value)

        return np.nan

    value = str(value).strip()

    if value.lower() in MISSING_VALUES:
        return np.nan

    # --------------------------------------------------------
    # Remove currency symbols
    # --------------------------------------------------------

    value = re.sub(
        r"[₹$€£¥]",
        "",
        value
    )

    # --------------------------------------------------------
    # Remove common currency codes
    # --------------------------------------------------------

    value = re.sub(
        r"\b(INR|USD|EUR|GBP|JPY)\b",
        "",
        value,
        flags=re.IGNORECASE
    )

    # --------------------------------------------------------
    # Remove commas
    # --------------------------------------------------------

    value = value.replace(",", "")

    # --------------------------------------------------------
    # Remove spaces
    # --------------------------------------------------------

    value = value.replace(" ", "")

    # --------------------------------------------------------
    # Handle percentages
    # --------------------------------------------------------

    if value.endswith("%"):

        value = value[:-1]

    # --------------------------------------------------------
    # Keep only valid numeric characters
    # --------------------------------------------------------

    value = re.sub(
        r"[^0-9.\-]",
        "",
        value
    )

    if value in {
        "",
        "-",
        ".",
        "-.",
    }:

        return np.nan

    try:

        return float(value)

    except ValueError:

        return np.nan


# ============================================================
# NORMALIZE DATE
# ============================================================

def normalize_date(value):
    """
    Normalize a single date value.

    Supports:
        2026-01-15
        15/02/2026
        15-02-2026
        March 10, 2026
        10 March 2026

    Invalid or empty values become NaT.
    """

    # Handle empty values
    if pd.isna(value):
        return pd.NaT

    value = str(value).strip()

    if value == "" or value.upper() in {
        "N/A",
        "NA",
        "NULL",
        "NONE",
        "-"
    }:
        return pd.NaT

    # --------------------------------------------------------
    # First attempt
    # --------------------------------------------------------

    result = pd.to_datetime(
        value,
        errors="coerce",
        format="mixed"
    )

    # --------------------------------------------------------
    # Second attempt
    # --------------------------------------------------------

    if pd.isna(result):

        result = pd.to_datetime(
            value,
            errors="coerce",
            dayfirst=True,
            format="mixed"
        )

    return result


# ============================================================
# NORMALIZE NUMERICAL COLUMNS
# ============================================================

def normalize_numeric_columns(df):

    df = df.copy()

    for column in NUMERICAL_FIELDS:

        if column in df.columns:

            df[column] = df[column].apply(
                normalize_numeric
            )

    return df


# ============================================================
# NORMALIZE DATE COLUMNS
# ============================================================

def normalize_date_columns(df):

    df = df.copy()

    for column in DATE_FIELDS:

        if column in df.columns:

            df[column] = df[column].apply(
                normalize_date
            )

    return df


# ============================================================
# NORMALIZE CATEGORICAL / STRING COLUMNS
# ============================================================

def normalize_string_columns(df):

    df = df.copy()

    for column in df.columns:

        # Don't touch numerical columns
        if column in NUMERICAL_FIELDS:
            continue

        # Don't touch date columns
        if column in DATE_FIELDS:
            continue

        df[column] = df[column].apply(
            normalize_string
        )

    return df


# ============================================================
# COMPLETE NORMALIZATION PIPELINE
# ============================================================

def normalize_dataframe(df):

    print(
        "\n========== DATA NORMALIZATION =========="
    )

    df = df.copy()

    # --------------------------------------------------------
    # Numeric normalization
    # --------------------------------------------------------

    print(
        "\nNormalizing numerical fields..."
    )

    df = normalize_numeric_columns(
        df
    )

    # --------------------------------------------------------
    # Date normalization
    # --------------------------------------------------------

    print(
        "Normalizing date fields..."
    )

    df = normalize_date_columns(
        df
    )

    # --------------------------------------------------------
    # String normalization
    # --------------------------------------------------------

    print(
        "Normalizing categorical fields..."
    )

    df = normalize_string_columns(
        df
    )

    print(
        "Normalization completed."
    )

    return df


# ============================================================
# DATA QUALITY SUMMARY
# ============================================================

def get_normalization_summary(df):

    summary = {

        "rows": len(df),

        "columns": len(df.columns),

        "missing_values": int(
            df.isnull().sum().sum()
        ),

        "duplicate_rows": int(
            df.duplicated().sum()
        ),

    }

    return summary


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_data = {

        "Billed_Amount": [
            "₹20,000",
            "15,500 INR",
            "$12,300",
            "10,000",
            "N/A",
        ],

        "Tax_Billed": [
            "18%",
            "₹2,500",
            "1,800 INR",
            "N/A",
            "900",
        ],

        "Actual_Usage": [
            "1,500",
            "2000",
            "1,250 units",
            "N/A",
            "900",
        ],

        "Customer_Name": [
            "  ABC Technologies  ",
            "XYZ Ltd",
            "   Global Corp   ",
            "N/A",
            "Test Company",
        ],

        "Invoice_Date": [
            "2026-01-15",
            "15/02/2026",
            "March 10, 2026",
            "N/A",
            "2026-05-20",
        ],
    }

    df = pd.DataFrame(
        test_data
    )

    print(
        "=" * 70
    )

    print(
        "NORMALIZER TEST"
    )

    print(
        "=" * 70
    )

    print(
        "\nBEFORE NORMALIZATION:\n"
    )

    print(df)

    normalized = normalize_dataframe(
        df
    )

    print(
        "\nAFTER NORMALIZATION:\n"
    )

    print(normalized)

    print(
        "\nDATA TYPES:\n"
    )

    print(
        normalized.dtypes
    )

    print(
        "\nSUMMARY:\n"
    )

    print(
        get_normalization_summary(
            normalized
        )
    )