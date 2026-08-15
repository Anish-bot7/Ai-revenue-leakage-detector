"""
Date Feature Builder

Converts Invoice_Date and Due_Date into the date features
expected by the trained ML preprocessor.

Supported date formats include:

    2026-01-15
    15/02/2026
    15-02-2026
    March 10, 2026
    10 March 2026
"""

import pandas as pd
import numpy as np


# ============================================================
# FLEXIBLE DATE PARSER
# ============================================================

def parse_flexible_dates(series):
    """
    Parse common invoice date formats safely.

    Examples:
        2026-01-15
        15/02/2026
        15-02-2026
        March 10, 2026
        10 March 2026
    """

    # --------------------------------------------------------
    # First attempt
    # --------------------------------------------------------

    result = pd.to_datetime(
        series,
        errors="coerce",
        format="mixed"
    )

    # --------------------------------------------------------
    # Retry failed values using day-first parsing
    # --------------------------------------------------------

    failed = result.isna()

    if failed.any():

        result.loc[failed] = pd.to_datetime(
            series.loc[failed],
            errors="coerce",
            dayfirst=True,
            format="mixed"
        )

    return result


# ============================================================
# BUILD DATE FEATURES
# ============================================================

def build_date_features(df):

    print(
        "\n========== DATE FEATURE ENGINEERING =========="
    )

    df = df.copy()

    # ========================================================
    # INVOICE DATE
    # ========================================================

    if "Invoice_Date" in df.columns:

        invoice_date = parse_flexible_dates(
            df["Invoice_Date"]
        )

        # Year
        df["Invoice_Year"] = (
            invoice_date.dt.year
        )

        # Month
        df["Invoice_Month"] = (
            invoice_date.dt.month
        )

        # Day
        df["Invoice_Day"] = (
            invoice_date.dt.day
        )

        # Monday = 0
        # Sunday = 6
        df["Invoice_DayOfWeek"] = (
            invoice_date.dt.dayofweek
        )

        # Quarter
        df["Invoice_Quarter"] = (
            invoice_date.dt.quarter
        )

    else:

        df["Invoice_Year"] = np.nan

        df["Invoice_Month"] = np.nan

        df["Invoice_Day"] = np.nan

        df["Invoice_DayOfWeek"] = np.nan

        df["Invoice_Quarter"] = np.nan

    # ========================================================
    # DUE DATE
    # ========================================================

    if "Due_Date" in df.columns:

        due_date = parse_flexible_dates(
            df["Due_Date"]
        )

        # Year
        df["Due_Year"] = (
            due_date.dt.year
        )

        # Month
        df["Due_Month"] = (
            due_date.dt.month
        )

        # Day
        df["Due_Day"] = (
            due_date.dt.day
        )

    else:

        df["Due_Year"] = np.nan

        df["Due_Month"] = np.nan

        df["Due_Day"] = np.nan

    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    print(
        "Date feature engineering completed."
    )

    return df


# ============================================================
# CHECK DATE FEATURE AVAILABILITY
# ============================================================

def get_date_feature_status(df):

    features = [

        "Invoice_Year",
        "Invoice_Month",
        "Invoice_Day",
        "Invoice_DayOfWeek",
        "Invoice_Quarter",

        "Due_Year",
        "Due_Month",
        "Due_Day",

    ]

    available = []

    unavailable = []

    for feature in features:

        if feature in df.columns:

            if df[feature].notna().any():

                available.append(
                    feature
                )

            else:

                unavailable.append(
                    feature
                )

        else:

            unavailable.append(
                feature
            )

    return available, unavailable


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 70
    )

    print(
        "DATE FEATURE TEST"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Test different date formats
    # --------------------------------------------------------

    test_data = pd.DataFrame({

        "Invoice_Date": [

            "2026-01-15",

            "15/02/2026",

            "March 10, 2026",

            "10 March 2026",

            "15-05-2026",

        ],

        "Due_Date": [

            "2026-02-15",

            "15/03/2026",

            "2026-04-10",

            "10 April 2026",

            "15-06-2026",

        ],

    })

    # --------------------------------------------------------
    # Display original data
    # --------------------------------------------------------

    print(
        "\nBEFORE:"
    )

    print(
        test_data
    )

    # --------------------------------------------------------
    # Build date features
    # --------------------------------------------------------

    result = build_date_features(
        test_data
    )

    # --------------------------------------------------------
    # Display generated features
    # --------------------------------------------------------

    print(
        "\nAFTER:"
    )

    print(
        result
    )

    # --------------------------------------------------------
    # Check availability
    # --------------------------------------------------------

    available, unavailable = (
        get_date_feature_status(
            result
        )
    )

    # --------------------------------------------------------
    # Available
    # --------------------------------------------------------

    print(
        "\nAvailable:"
    )

    for feature in available:

        print(
            f"  ✓ {feature}"
        )

    # --------------------------------------------------------
    # Unavailable
    # --------------------------------------------------------

    print(
        "\nUnavailable:"
    )

    for feature in unavailable:

        print(
            f"  - {feature}"
        )

    # --------------------------------------------------------
    # Verify expected values
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DATE PARSING VERIFICATION"
    )

    print(
        "=" * 70
    )

    print(
        "\nInvoice years:"
    )

    print(
        result["Invoice_Year"].tolist()
    )

    print(
        "\nInvoice months:"
    )

    print(
        result["Invoice_Month"].tolist()
    )

    print(
        "\nInvoice days:"
    )

    print(
        result["Invoice_Day"].tolist()
    )

    print(
        "\nDue years:"
    )

    print(
        result["Due_Year"].tolist()
    )

    print(
        "\nDue months:"
    )

    print(
        result["Due_Month"].tolist()
    )

    print(
        "\nDue days:"
    )

    print(
        result["Due_Day"].tolist()
    )