"""
Complete input preprocessing pipeline.

Flow:

CSV / Excel
    ↓
Column Mapping
    ↓
Normalization
    ↓
Validation
    ↓
Feature Engineering
    ↓
Date Feature Engineering
    ↓
Clean canonical DataFrame
"""

import pandas as pd

from .column_mapper import apply_column_mapping
from .normalizer import normalize_dataframe
from .validator import validate_uploaded_data
from .feature_builder import build_features
from .date_features import build_date_features


# ============================================================
# LOAD FILE
# ============================================================

def load_input_file(file_path):
    """
    Load CSV or Excel file.
    Supports:
        .csv
        .xlsx
        .xls
    """

    file_path = str(file_path)

    if file_path.lower().endswith(".csv"):

        df = pd.read_csv(
            file_path,
            keep_default_na=False
        )

    elif file_path.lower().endswith(
        (".xlsx", ".xls")
    ):

        df = pd.read_excel(
            file_path
        )

    else:

        raise ValueError(
            "Unsupported file format. "
            "Please upload CSV or Excel."
        )

    return df


# ============================================================
# RUN COMPLETE PIPELINE
# ============================================================

def process_uploaded_file(
    file_path,
    fuzzy_threshold=0.88
):

    print("\n")
    print("=" * 70)
    print(" REVENUE LEAKAGE INPUT PIPELINE")
    print("=" * 70)

    # ========================================================
    # STEP 1: LOAD
    # ========================================================

    print("\n[1/6] Loading file...")

    df = load_input_file(
        file_path
    )

    print(
        f"Loaded successfully: "
        f"{df.shape}"
    )

    # ========================================================
    # STEP 2: COLUMN MAPPING
    # ========================================================

    print("\n[2/6] Mapping columns...")

    (
        mapped_df,
        mapping,
        confidence
    ) = apply_column_mapping(
        df,
        fuzzy_threshold
    )

    print("\nColumn mappings:")

    for original, canonical in mapping.items():

        print(
            f"  {original:30} -> "
            f"{canonical:25} "
            f"({confidence[original]:.2f})"
        )

    # ========================================================
    # STEP 3: NORMALIZATION
    # ========================================================

    print("\n[3/6] Normalizing data...")

    normalized_df = normalize_dataframe(
        mapped_df
    )

    # ========================================================
    # STEP 4: VALIDATION
    # ========================================================

    print("\n[4/6] Validating data...")

    validation = validate_uploaded_data(
        normalized_df
    )

    # ========================================================
    # STOP IF INVALID
    # ========================================================

    if not validation.valid:

        print("\n")
        print("=" * 70)
        print(" ❌ FILE REJECTED")
        print("=" * 70)

        return {

            "success": False,

            "data": None,

            "mapping": mapping,

            "confidence": confidence,

            "validation":
                validation.to_dict(),

        }

    # ========================================================
    # STEP 5: FEATURE ENGINEERING
    # ========================================================

    print("\n[5/6] Building features...")

    featured_df = build_features(
        normalized_df
    )

    print(
        "Feature engineering completed."
    )

    # ========================================================
    # STEP 6: DATE FEATURES
    # ========================================================

    print(
        "\n[6/6] Building date features..."
    )

    featured_df = build_date_features(
        featured_df
    )

    print(
        "Date feature engineering completed."
    )

    # ========================================================
    # SUCCESS
    # ========================================================

    print("\n")
    print("=" * 70)
    print(" ✅ FILE READY FOR ML PROCESSING")
    print("=" * 70)

    print(
        f"\nRows: "
        f"{len(featured_df)}"
    )

    print(
        f"Columns: "
        f"{len(featured_df.columns)}"
    )

    print(
        "\nGenerated feature columns:"
    )

    generated_features = [
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
        "Invoice_Year",
        "Invoice_Month",
        "Invoice_Day",
        "Invoice_DayOfWeek",
        "Invoice_Quarter",
        "Due_Year",
        "Due_Month",
        "Due_Day"
    ]

    for feature in generated_features:

        if feature in featured_df.columns:

            print(
                f"  ✓ {feature}"
            )

    return {

        "success": True,

        "data": featured_df,

        "mapping": mapping,

        "confidence": confidence,

        "validation":
            validation.to_dict(),

    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PIPELINE TEST"
    )

    print(
        "=" * 70
    )

    test_file = (
        "sample_inputs/company_test.csv"
    )

    try:

        result = process_uploaded_file(
            test_file
        )

        if result["success"]:

            print(
                "\nFirst rows after complete preprocessing:\n"
            )

            print(
                result["data"].head()
            )

            print(
                "\nFinal columns:"
            )

            print(
                result["data"].columns.tolist()
            )

    except FileNotFoundError:

        print(
            "\nTest file not found:"
        )

        print(
            test_file
        )

        print(
            "\nCreate this file first to run "
            "the pipeline test."
        )

    except Exception as e:

        print(
            "\nPipeline error:"
        )

        print(
            type(e).__name__,
            ":",
            str(e)
        )