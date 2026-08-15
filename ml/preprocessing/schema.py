"""
Canonical schema for the AI Revenue Leakage Detection system.

This file defines the business-level fields that our ML pipeline
understands, regardless of what column names the user's CSV/Excel uses.
"""


# ============================================================
# REQUIRED FIELDS
# ============================================================

REQUIRED_FIELDS = [

    "Expected_Amount",
    "Billed_Amount",

]


# ============================================================
# HIGH-VALUE OPTIONAL FIELDS
# ============================================================

IMPORTANT_OPTIONAL_FIELDS = [

    "Actual_Usage",
    "Billed_Usage",

    "Unit_Price",
    "Contract_Price",

    "Tax_Expected",
    "Tax_Billed",

    "Discount_Allowed",
    "Discount_Applied",
    "Discount_Amount",

]


# ============================================================
# FULL CANONICAL SCHEMA
# ============================================================

CANONICAL_FIELDS = [

    # --------------------------------------------------------
    # Invoice
    # --------------------------------------------------------

    "Invoice_ID",

    # --------------------------------------------------------
    # Customer
    # --------------------------------------------------------

    "Customer_ID",
    "Customer_Name",
    "Customer_Type",
    "Industry",
    "Region",
    "Country",
    "City",
    "Email",
    "Phone",
    "Registration_Date",
    "Account_Status",
    "Payment_Terms",
    "Credit_Limit",
    "Account_Manager",

    # --------------------------------------------------------
    # Contract
    # --------------------------------------------------------

    "Contract_ID",
    "Service_Type",
    "Plan_Name",
    "Unit_Price",
    "Estimated_Usage",
    "Contract_Price",
    "Discount_Allowed",
    "Tax_Rate",
    "Billing_Cycle",
    "Contract_Duration",
    "SLA_Level",

    # --------------------------------------------------------
    # Usage
    # --------------------------------------------------------

    "Billing_Month",
    "Billing_Year",
    "Actual_Usage",
    "Billed_Usage",
    "Peak_Usage",
    "OffPeak_Usage",
    "Expected_Amount",

    # --------------------------------------------------------
    # Invoice Details
    # --------------------------------------------------------

    "Invoice_Date",
    "Due_Date",
    "Currency",
    "Invoice_Status",
    "Payment_Status",
    "Invoice_Version",

    "Tax_Expected",
    "Tax_Billed",

    "Discount_Applied",
    "Discount_Amount",

    "Billed_Amount",

]


# ============================================================
# TARGET / TRAINING-ONLY FIELDS
# ============================================================

TARGET_FIELDS = [

    "Leakage",
    "Leakage_Type",
    "Leakage_Amount",

]


# ============================================================
# LEAKAGE INDICATOR FIELDS
# ============================================================

LEAKAGE_INDICATOR_FIELDS = [

    "Missing_GST",
    "Usage_Mismatch",
    "Extra_Discount",
    "Wrong_Pricing",
    "Missing_Charge",
    "Duplicate_Invoice",
    "Contract_Violation",
    "Late_Billing",

]


# ============================================================
# ALL TRAINING COLUMNS
# ============================================================

TRAINING_FIELDS = (
    CANONICAL_FIELDS
    + LEAKAGE_INDICATOR_FIELDS
    + TARGET_FIELDS
)


# ============================================================
# FIELD TYPES
# ============================================================

NUMERICAL_FIELDS = [

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


CATEGORICAL_FIELDS = [

    "Customer_Type",
    "Industry",
    "Region",
    "Country",
    "City",
    "Account_Status",
    "Payment_Terms",
    "Account_Manager",

    "Service_Type",
    "Plan_Name",
    "Billing_Cycle",
    "SLA_Level",

    "Currency",
    "Invoice_Status",
    "Payment_Status",

]


DATE_FIELDS = [

    "Registration_Date",
    "Invoice_Date",
    "Due_Date",

]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def is_required_field(field):

    return field in REQUIRED_FIELDS


def is_known_field(field):

    return field in CANONICAL_FIELDS


def is_target_field(field):

    return field in TARGET_FIELDS


def get_missing_required_fields(columns):

    columns = set(columns)

    return [
        field
        for field in REQUIRED_FIELDS
        if field not in columns
    ]


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CANONICAL REVENUE LEAKAGE SCHEMA")
    print("=" * 60)

    print("\nRequired fields:")

    for field in REQUIRED_FIELDS:
        print(f"  - {field}")

    print("\nImportant optional fields:")

    for field in IMPORTANT_OPTIONAL_FIELDS:
        print(f"  - {field}")

    print(
        f"\nTotal canonical fields: "
        f"{len(CANONICAL_FIELDS)}"
    )

    print(
        f"Total numerical fields: "
        f"{len(NUMERICAL_FIELDS)}"
    )

    print(
        f"Total categorical fields: "
        f"{len(CATEGORICAL_FIELDS)}"
    )

    print(
        f"Total date fields: "
        f"{len(DATE_FIELDS)}"
    )