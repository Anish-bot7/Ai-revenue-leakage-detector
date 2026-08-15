"""
Feature Builder

Converts canonical billing data into the engineered features
used by the revenue leakage ML model.
"""

import numpy as np
import pandas as pd


# ============================================================
# SAFE DIVISION
# ============================================================

def safe_divide(
    numerator,
    denominator
):

    denominator = denominator.replace(
        0,
        np.nan
    )

    result = numerator / denominator

    return result.replace(
        [np.inf, -np.inf],
        np.nan
    )


# ============================================================
# BUILD FEATURES
# ============================================================

def build_features(df):

    print(
        "\n========== FEATURE BUILDING =========="
    )

    df = df.copy()

    # ========================================================
    # BASIC NUMERIC COLUMNS
    # ========================================================

    numeric_defaults = [

        "Credit_Limit",
        "Unit_Price",
        "Estimated_Usage",
        "Contract_Price",
        "Discount_Allowed",
        "Tax_Rate",
        "Contract_Duration",

        "Billing_Month",
        "Billing_Year",

        "Actual_Usage",
        "Billed_Usage",

        "Peak_Usage",
        "OffPeak_Usage",

        "Expected_Amount",

        "Invoice_Version",

        "Tax_Expected",
        "Tax_Billed",

        "Discount_Applied",
        "Discount_Amount",

        "Billed_Amount",

    ]

    for column in numeric_defaults:

        if column not in df.columns:

            df[column] = np.nan

    # ========================================================
    # PAYMENT TERM DAYS
    # ========================================================

    if "Payment_Terms" in df.columns:

        df["Payment_Term_Days"] = (
            df["Payment_Terms"]
            .astype(str)
            .str.extract(
                r"(\d+)"
            )[0]
            .astype(float)
        )

    else:

        df["Payment_Term_Days"] = np.nan

    # ========================================================
    # CUSTOMER TENURE
    # ========================================================

    if "Registration_Date" in df.columns:

        registration_date = pd.to_datetime(
            df["Registration_Date"],
            errors="coerce"
        )

        reference_date = pd.Timestamp.today()

        df["Customer_Tenure_Days"] = (
            reference_date
            - registration_date
        ).dt.days

        df["Customer_Tenure_Years"] = (
            df["Customer_Tenure_Days"]
            / 365.25
        )

    else:

        df["Customer_Tenure_Days"] = np.nan

        df["Customer_Tenure_Years"] = np.nan

    # ========================================================
    # USAGE DIFFERENCE
    # ========================================================

    df["Usage_Difference"] = (

        df["Actual_Usage"]
        - df["Billed_Usage"]

    )

    # ========================================================
    # USAGE BILLING RATIO
    # ========================================================

    df["Usage_Billing_Ratio"] = safe_divide(

        df["Billed_Usage"],

        df["Actual_Usage"]

    )

    # ========================================================
    # AMOUNT DIFFERENCE
    # ========================================================

    df["Amount_Difference"] = (

        df["Expected_Amount"]
        - df["Billed_Amount"]

    )

    # ========================================================
    # BILLING RATIO
    # ========================================================

    df["Billing_Ratio"] = safe_divide(

        df["Billed_Amount"],

        df["Expected_Amount"]

    )

    # ========================================================
    # TAX DIFFERENCE
    # ========================================================

    df["Tax_Difference"] = (

        df["Tax_Expected"]
        - df["Tax_Billed"]

    )

    # ========================================================
    # TAX BILLING RATIO
    # ========================================================

    df["Tax_Billing_Ratio"] = safe_divide(

        df["Tax_Billed"],

        df["Tax_Expected"]

    )

    # ========================================================
    # DISCOUNT DIFFERENCE
    # ========================================================

    df["Discount_Difference"] = (

        df["Discount_Allowed"]
        - df["Discount_Applied"]

    )

    # ========================================================
    # DISCOUNT EXCEEDED
    # ========================================================

    df["Discount_Exceeded"] = (

        df["Discount_Applied"]
        > df["Discount_Allowed"]

    ).astype(int)

    # ========================================================
    # EXPECTED UNIT REVENUE
    # ========================================================

    df["Expected_Unit_Revenue"] = safe_divide(

        df["Expected_Amount"],

        df["Actual_Usage"]

    )

    # ========================================================
    # ACTUAL UNIT REVENUE
    # ========================================================

    df["Actual_Unit_Revenue"] = safe_divide(

        df["Billed_Amount"],

        df["Billed_Usage"]

    )

    # ========================================================
    # UNIT REVENUE DIFFERENCE
    # ========================================================

    df["Unit_Revenue_Difference"] = (

        df["Expected_Unit_Revenue"]
        - df["Actual_Unit_Revenue"]

    )

    # ========================================================
    # USAGE UTILIZATION
    # ========================================================

    df["Usage_Utilization"] = safe_divide(

        df["Actual_Usage"],

        df["Estimated_Usage"]

    )

    # ========================================================
    # CONTRACT AMOUNT DIFFERENCE
    # ========================================================

    df["Contract_Amount_Difference"] = (

        df["Contract_Price"]
        - df["Expected_Amount"]

    )

    # ========================================================
    # PEAK USAGE RATIO
    # ========================================================

    df["Peak_Usage_Ratio"] = safe_divide(

        df["Peak_Usage"],

        df["Actual_Usage"]

    )

    # ========================================================
    # OFF-PEAK USAGE RATIO
    # ========================================================

    df["OffPeak_Usage_Ratio"] = safe_divide(

        df["OffPeak_Usage"],

        df["Actual_Usage"]

    )

    # ========================================================
    # REPLACE INFINITE VALUES
    # ========================================================

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    print(
        "Feature engineering completed."
    )

    return df


# ============================================================
# FEATURE AVAILABILITY
# ============================================================

def get_feature_availability(df):

    engineered_features = [

        "Payment_Term_Days",
        "Customer_Tenure_Days",
        "Customer_Tenure_Years",

        "Usage_Difference",
        "Usage_Billing_Ratio",

        "Amount_Difference",
        "Billing_Ratio",

        "Tax_Difference",
        "Tax_Billing_Ratio",

        "Discount_Difference",
        "Discount_Exceeded",

        "Expected_Unit_Revenue",
        "Actual_Unit_Revenue",
        "Unit_Revenue_Difference",

        "Usage_Utilization",

        "Contract_Amount_Difference",

        "Peak_Usage_Ratio",
        "OffPeak_Usage_Ratio",

    ]

    available = []

    unavailable = []

    for feature in engineered_features:

        if feature in df.columns:

            if df[feature].notna().any():

                available.append(feature)

            else:

                unavailable.append(feature)

        else:

            unavailable.append(feature)

    return available, unavailable


# ============================================================
# DISPLAY FEATURE SUMMARY
# ============================================================

def display_feature_summary(df):

    available, unavailable = (
        get_feature_availability(df)
    )

    print(
        "\n========== FEATURE AVAILABILITY =========="
    )

    print(
        "\nAvailable engineered features:"
    )

    for feature in available:

        print(
            f"  ✓ {feature}"
        )

    print(
        "\nUnavailable engineered features:"
    )

    for feature in unavailable:

        print(
            f"  - {feature}"
        )

    print(
        f"\nAvailable: {len(available)}"
    )

    print(
        f"Unavailable: {len(unavailable)}"
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_data = pd.DataFrame({

        "Actual_Usage": [
            1500,
            2000,
            1250,
        ],

        "Billed_Usage": [
            1450,
            2000,
            1150,
        ],

        "Expected_Amount": [
            20000,
            25000,
            18000,
        ],

        "Billed_Amount": [
            19000,
            25000,
            16000,
        ],

        "Tax_Expected": [
            3600,
            4500,
            3240,
        ],

        "Tax_Billed": [
            3420,
            4500,
            2800,
        ],

        "Discount_Applied": [
            5,
            10,
            5,
        ],

        "Unit_Price": [
            10,
            10,
            15,
        ],

    })

    print(
        "=" * 70
    )

    print(
        "FEATURE BUILDER TEST"
    )

    print(
        "=" * 70
    )

    result = build_features(
        test_data
    )

    display_feature_summary(
        result
    )

    print(
        "\nGenerated features:\n"
    )

    print(
        result[
            [
                "Usage_Difference",
                "Usage_Billing_Ratio",
                "Amount_Difference",
                "Billing_Ratio",
                "Tax_Difference",
                "Tax_Billing_Ratio",
                "Expected_Unit_Revenue",
                "Actual_Unit_Revenue",
                "Unit_Revenue_Difference",
            ]
        ]
    )