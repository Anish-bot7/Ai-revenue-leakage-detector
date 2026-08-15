"""
Model Input Builder

Prepares flexible user-uploaded billing data for the
existing trained ML model.
"""

import numpy as np
import pandas as pd

from .feature_builder import build_features


# ============================================================
# MODEL FEATURES
# ============================================================

ENGINEERED_FEATURES = [

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


# ============================================================
# MODEL INPUT PREPARATION
# ============================================================

def prepare_model_input(df):

    print(
        "\n========== MODEL INPUT PREPARATION =========="
    )

    # --------------------------------------------------------
    # Copy dataframe
    # --------------------------------------------------------

    df = df.copy()

    # --------------------------------------------------------
    # Build engineered features
    # --------------------------------------------------------

    df = build_features(
        df
    )

    # --------------------------------------------------------
    # Check feature availability
    # --------------------------------------------------------

    available_features = []

    missing_features = []

    for feature in ENGINEERED_FEATURES:

        if feature in df.columns:

            if df[feature].notna().any():

                available_features.append(
                    feature
                )

            else:

                missing_features.append(
                    feature
                )

        else:

            missing_features.append(
                feature
            )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print(
        "\nAvailable engineered features:"
    )

    for feature in available_features:

        print(
            f"  ✓ {feature}"
        )

    print(
        "\nUnavailable engineered features:"
    )

    for feature in missing_features:

        print(
            f"  - {feature}"
        )

    # --------------------------------------------------------
    # Feature coverage
    # --------------------------------------------------------

    total_features = len(
        ENGINEERED_FEATURES
    )

    available_count = len(
        available_features
    )

    coverage = (
        available_count
        / total_features
    )

    print(
        f"\nFeature coverage: "
        f"{available_count}/{total_features} "
        f"({coverage * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # Data quality score
    # --------------------------------------------------------

    data_quality_score = round(
        coverage * 100,
        2
    )

    # --------------------------------------------------------
    # Create model dataframe
    # --------------------------------------------------------

    model_df = df.copy()

    # --------------------------------------------------------
    # Add missing engineered features
    # --------------------------------------------------------

    for feature in missing_features:

        model_df[feature] = np.nan

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {

        "data": model_df,

        "available_features":
            available_features,

        "missing_features":
            missing_features,

        "feature_coverage":
            round(
                coverage,
                4
            ),

        "data_quality_score":
            data_quality_score,

    }


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

    result = prepare_model_input(
        test_data
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MODEL INPUT TEST"
    )

    print(
        "=" * 70
    )

    print(
        "\nData quality score:",
        result["data_quality_score"]
    )

    print(
        "\nModel input shape:",
        result["data"].shape
    )