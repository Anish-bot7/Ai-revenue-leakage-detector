"""
Validation layer for the Revenue Leakage Detection system.

Checks whether an uploaded CSV/Excel file contains enough
information for reliable revenue leakage analysis.
"""

import pandas as pd

from .schema import (
    REQUIRED_FIELDS,
    IMPORTANT_OPTIONAL_FIELDS,
)


# ============================================================
# VALIDATION RESULT
# ============================================================

class ValidationResult:

    def __init__(
        self,
        valid,
        missing_required=None,
        available_fields=None,
        optional_missing=None,
        warnings=None,
    ):

        self.valid = valid

        self.missing_required = (
            missing_required or []
        )

        self.available_fields = (
            available_fields or []
        )

        self.optional_missing = (
            optional_missing or []
        )

        self.warnings = (
            warnings or []
        )

    # --------------------------------------------------------
    # Convert result to dictionary
    # --------------------------------------------------------

    def to_dict(self):

        return {

            "valid": self.valid,

            "missing_required":
                self.missing_required,

            "available_fields":
                self.available_fields,

            "optional_missing":
                self.optional_missing,

            "warnings":
                self.warnings,

        }

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    def display(self):

        print(
            "\n"
            + "=" * 70
        )

        print(
            "DATA VALIDATION RESULT"
        )

        print(
            "=" * 70
        )

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        if self.valid:

            print(
                "\n✅ FILE CAN BE ANALYZED"
            )

        else:

            print(
                "\n❌ FILE CANNOT BE ANALYZED"
            )

        # ----------------------------------------------------
        # Missing required fields
        # ----------------------------------------------------

        if self.missing_required:

            print(
                "\nMissing required fields:"
            )

            for field in self.missing_required:

                print(
                    f"  ❌ {field}"
                )

        # ----------------------------------------------------
        # Available fields
        # ----------------------------------------------------

        print(
            "\nAvailable canonical fields:"
        )

        for field in self.available_fields:

            print(
                f"  ✓ {field}"
            )

        # ----------------------------------------------------
        # Optional fields
        # ----------------------------------------------------

        if self.optional_missing:

            print(
                "\nOptional fields not available:"
            )

            for field in self.optional_missing:

                print(
                    f"  - {field}"
                )

        # ----------------------------------------------------
        # Warnings
        # ----------------------------------------------------

        if self.warnings:

            print(
                "\nWarnings:"
            )

            for warning in self.warnings:

                print(
                    f"  ⚠ {warning}"
                )


# ============================================================
# CHECK DATAFRAME
# ============================================================

def validate_dataframe(df):

    # --------------------------------------------------------
    # Check DataFrame
    # --------------------------------------------------------

    if not isinstance(
        df,
        pd.DataFrame
    ):

        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    # --------------------------------------------------------
    # Check empty file
    # --------------------------------------------------------

    if df.empty:

        return ValidationResult(

            valid=False,

            warnings=[
                "Uploaded file contains no records."
            ],

        )

    # --------------------------------------------------------
    # Get columns
    # --------------------------------------------------------

    columns = set(
        df.columns
    )

    # --------------------------------------------------------
    # Required fields
    # --------------------------------------------------------

    missing_required = [

        field

        for field in REQUIRED_FIELDS

        if field not in columns

    ]

    # --------------------------------------------------------
    # Available fields
    # --------------------------------------------------------

    available_fields = [

        field

        for field in df.columns

    ]

    # --------------------------------------------------------
    # Missing optional fields
    # --------------------------------------------------------

    optional_missing = [

        field

        for field in IMPORTANT_OPTIONAL_FIELDS

        if field not in columns

    ]

    # --------------------------------------------------------
    # Warnings
    # --------------------------------------------------------

    warnings = []

    # Too few columns

    if len(df.columns) < 3:

        warnings.append(
            "The uploaded file contains very few columns."
        )

    # Very small dataset

    if len(df) < 5:

        warnings.append(
            "The uploaded file contains fewer than 5 records."
        )

    # Missing usage information

    if (
        "Actual_Usage" not in columns
        and
        "Billed_Usage" not in columns
    ):

        warnings.append(
            "Usage information is unavailable. "
            "Some leakage patterns may not be detectable."
        )

    # Missing tax information

    if (
        "Tax_Expected" not in columns
        and
        "Tax_Billed" not in columns
    ):

        warnings.append(
            "Tax information is unavailable. "
            "GST/tax leakage may not be detectable."
        )

    # Missing contract information

    if (
        "Contract_Price" not in columns
        and
        "Unit_Price" not in columns
    ):

        warnings.append(
            "Contract pricing information is unavailable."
        )

    # --------------------------------------------------------
    # Determine validity
    # --------------------------------------------------------

    valid = (
        len(missing_required) == 0
    )

    # --------------------------------------------------------
    # Create result
    # --------------------------------------------------------

    return ValidationResult(

        valid=valid,

        missing_required=
            missing_required,

        available_fields=
            available_fields,

        optional_missing=
            optional_missing,

        warnings=
            warnings,

    )


# ============================================================
# VALIDATE AFTER COLUMN MAPPING
# ============================================================

def validate_uploaded_data(df):

    """
    Main validation function.

    This function is called AFTER column mapping and
    normalization.
    """

    print(
        "\n========== DATA VALIDATION =========="
    )

    result = validate_dataframe(
        df
    )

    if result.valid:

        print(
            "\n[PASS] Required billing information found."
        )

        print(
            "File is eligible for prediction."
        )

    else:

        print(
            "\n❌ Required billing information missing."
        )

        print(
            "Prediction should NOT be performed."
        )

    return result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 70
    )

    print(
        "VALIDATOR TEST"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # GOOD FILE
    # --------------------------------------------------------

    good_data = pd.DataFrame({

        "Expected_Amount": [
            10000,
            20000,
            30000,
        ],

        "Billed_Amount": [
            9500,
            20000,
            25000,
        ],

        "Actual_Usage": [
            1000,
            2000,
            3000,
        ],

        "Billed_Usage": [
            950,
            2000,
            2500,
        ],

    })

    print(
        "\n\nTEST 1: VALID BILLING FILE"
    )

    result = validate_dataframe(
        good_data
    )

    result.display()

    # --------------------------------------------------------
    # BAD FILE
    # --------------------------------------------------------

    bad_data = pd.DataFrame({

        "Customer_Name": [
            "ABC Ltd",
            "XYZ Ltd",
        ],

        "Industry": [
            "IT",
            "Finance",
        ],

        "Region": [
            "South",
            "North",
        ],

    })

    print(
        "\n\nTEST 2: INVALID FILE"
    )

    result = validate_dataframe(
        bad_data
    )

    result.display()

    # --------------------------------------------------------
    # EMPTY FILE
    # --------------------------------------------------------

    empty_data = pd.DataFrame()

    print(
        "\n\nTEST 3: EMPTY FILE"
    )

    result = validate_dataframe(
        empty_data
    )

    result.display()