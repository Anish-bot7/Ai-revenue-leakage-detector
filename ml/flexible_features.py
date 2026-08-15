"""
Feature definition for the flexible revenue leakage model.

These features are selected because they can realistically
be obtained from different billing CSV/Excel formats.
"""

FLEXIBLE_NUMERIC_FEATURES = [

    # Raw billing information
    "Actual_Usage",
    "Billed_Usage",

    "Expected_Amount",
    "Billed_Amount",

    "Unit_Price",

    "Tax_Expected",
    "Tax_Billed",

    "Discount_Allowed",
    "Discount_Applied",

    "Discount_Amount",

    # Engineered billing discrepancies
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

]


# ============================================================
# OPTIONAL FEATURES
# ============================================================

OPTIONAL_NUMERIC_FEATURES = [

    "Contract_Price",

    "Estimated_Usage",

    "Tax_Rate",

    "Invoice_Version",

]


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

FLEXIBLE_CATEGORICAL_FEATURES = [

    "Customer_Type",

    "Industry",

    "Region",

    "Service_Type",

    "Billing_Cycle",

    "Currency",

    "Invoice_Status",

]


# ============================================================
# DATE FEATURES
# ============================================================

FLEXIBLE_DATE_FEATURES = [

    "Invoice_Year",

    "Invoice_Month",

    "Invoice_Day",

    "Invoice_DayOfWeek",

    "Invoice_Quarter",

]


# ============================================================
# ALL FEATURES
# ============================================================

FLEXIBLE_FEATURES = (

    FLEXIBLE_NUMERIC_FEATURES

    + OPTIONAL_NUMERIC_FEATURES

    + FLEXIBLE_CATEGORICAL_FEATURES

    + FLEXIBLE_DATE_FEATURES

)


# ============================================================
# DISPLAY
# ============================================================

if __name__ == "__main__":

    print("=" * 70)

    print(
        "FLEXIBLE MODEL FEATURE DEFINITION"
    )

    print("=" * 70)

    print(
        "\nNumerical features:",
        len(FLEXIBLE_NUMERIC_FEATURES)
    )

    for feature in FLEXIBLE_NUMERIC_FEATURES:

        print(
            f"  ✓ {feature}"
        )

    print(
        "\nOptional features:",
        len(OPTIONAL_NUMERIC_FEATURES)
    )

    for feature in OPTIONAL_NUMERIC_FEATURES:

        print(
            f"  + {feature}"
        )

    print(
        "\nCategorical features:",
        len(FLEXIBLE_CATEGORICAL_FEATURES)
    )

    for feature in FLEXIBLE_CATEGORICAL_FEATURES:

        print(
            f"  ✓ {feature}"
        )

    print(
        "\nDate features:",
        len(FLEXIBLE_DATE_FEATURES)
    )

    for feature in FLEXIBLE_DATE_FEATURES:

        print(
            f"  ✓ {feature}"
        )

    print(
        "\nTotal flexible features:",
        len(FLEXIBLE_FEATURES)
    )