"""
Automatically maps different CSV/Excel column names
to the canonical revenue leakage schema.
"""

import re

from difflib import SequenceMatcher

from .schema import (
    CANONICAL_FIELDS,
    REQUIRED_FIELDS,
)


# ============================================================
# COLUMN ALIASES
# ============================================================

COLUMN_ALIASES = {

    # --------------------------------------------------------
    # Invoice
    # --------------------------------------------------------

    "Invoice_ID": [
        "invoice_id",
        "invoice",
        "invoice_number",
        "invoice_no",
        "invoice_num",
        "inv_id",
        "inv_no",
        "bill_number",
        "bill_no",
    ],

    # --------------------------------------------------------
    # Customer
    # --------------------------------------------------------

    "Customer_ID": [
        "customer_id",
        "customer_number",
        "client_id",
        "account_id",
        "customer_no",
    ],

    "Customer_Name": [
        "customer_name",
        "customer",
        "client",
        "client_name",
        "company",
        "company_name",
        "account_name",
    ],

    "Customer_Type": [
        "customer_type",
        "client_type",
        "account_type",
        "customer_segment",
        "segment",
    ],

    "Industry": [
        "industry",
        "business_sector",
        "sector",
        "industry_type",
    ],

    "Region": [
        "region",
        "territory",
        "sales_region",
        "geographical_region",
    ],

    "Country": [
        "country",
        "country_name",
        "nation",
    ],

    "City": [
        "city",
        "city_name",
        "location_city",
    ],

    "Email": [
        "email",
        "email_address",
        "customer_email",
    ],

    "Phone": [
        "phone",
        "phone_number",
        "mobile",
        "mobile_number",
        "contact_number",
    ],

    "Registration_Date": [
        "registration_date",
        "customer_since",
        "account_start_date",
        "signup_date",
    ],

    "Account_Status": [
        "account_status",
        "customer_status",
        "status",
        "account_state",
    ],

    "Payment_Terms": [
        "payment_terms",
        "payment_term",
        "payment_condition",
        "credit_terms",
    ],

    "Credit_Limit": [
        "credit_limit",
        "credit_amount",
        "account_credit_limit",
    ],

    "Account_Manager": [
        "account_manager",
        "relationship_manager",
        "sales_manager",
        "customer_manager",
    ],

    # --------------------------------------------------------
    # Contract
    # --------------------------------------------------------

    "Contract_ID": [
        "contract_id",
        "contract_number",
        "contract_no",
        "agreement_id",
        "agreement_number",
    ],

    "Service_Type": [
        "service_type",
        "service",
        "service_name",
        "product",
        "product_type",
        "product_name",
    ],

    "Plan_Name": [
        "plan_name",
        "plan",
        "subscription_plan",
        "package",
        "package_name",
    ],

    "Unit_Price": [
        "unit_price",
        "unit_rate",
        "price_per_unit",
        "rate_per_unit",
        "unit_cost",
        "rate",
        "price",
    ],

    "Estimated_Usage": [
        "estimated_usage",
        "expected_usage",
        "planned_usage",
        "forecast_usage",
        "estimated_consumption",
    ],

    "Contract_Price": [
        "contract_price",
        "contract_amount",
        "contract_value",
        "agreement_value",
        "contract_total",
    ],

    "Discount_Allowed": [
        "discount_allowed",
        "allowed_discount",
        "maximum_discount",
        "max_discount",
        "contract_discount",
        "permitted_discount",
    ],

    "Tax_Rate": [
        "tax_rate",
        "tax_percent",
        "tax_percentage",
        "gst_rate",
        "vat_rate",
    ],

    "Billing_Cycle": [
        "billing_cycle",
        "billing_frequency",
        "invoice_frequency",
        "billing_period",
    ],

    "Contract_Duration": [
        "contract_duration",
        "contract_length",
        "agreement_duration",
        "duration_months",
    ],

    "SLA_Level": [
        "sla_level",
        "sla",
        "service_level",
        "service_level_agreement",
    ],

    # --------------------------------------------------------
    # Usage
    # --------------------------------------------------------

    "Billing_Month": [
        "billing_month",
        "invoice_month",
        "month",
        "billing_period_month",
    ],

    "Billing_Year": [
        "billing_year",
        "invoice_year",
        "year",
        "billing_period_year",
    ],

    "Actual_Usage": [
        "actual_usage",
        "actual use",
        "usage",
        "consumption",
        "actual_consumption",
        "actual_units",
        "units_consumed",
        "quantity_used",
        "consumed_quantity",
        "consumed_qty",
    ],

    "Billed_Usage": [
        "billed_usage",
        "billed use",
        "billed_units",
        "billable_units",
        "charged_units",
        "quantity_billed",
        "billed_quantity",
        "invoice_quantity",
        "charged_quantity",
    ],

    "Peak_Usage": [
        "peak_usage",
        "peak_consumption",
        "peak_units",
        "peak_quantity",
    ],

    "OffPeak_Usage": [
        "offpeak_usage",
        "off_peak_usage",
        "offpeak_consumption",
        "off_peak_consumption",
    ],

    "Expected_Amount": [
        "expected_amount",
        "expected_total",
        "expected_revenue",
        "expected_charge",
        "expected_bill",
        "expected_invoice",
        "expected_value",
        "contract_value",
        "expected_total_amount",
        "should_be_billed",
        "should_bill",
    ],

    # --------------------------------------------------------
    # Invoice
    # --------------------------------------------------------

    "Invoice_Date": [
        "invoice_date",
        "billing_date",
        "bill_date",
        "issue_date",
        "invoice_issued_date",
    ],

    "Due_Date": [
        "due_date",
        "payment_due_date",
        "invoice_due_date",
        "due",
    ],

    "Currency": [
        "currency",
        "currency_code",
        "currency_type",
    ],

    "Invoice_Status": [
        "invoice_status",
        "bill_status",
        "invoice_state",
    ],

    "Payment_Status": [
        "payment_status",
        "payment_state",
        "paid_status",
        "payment_condition",
    ],

    "Invoice_Version": [
        "invoice_version",
        "version",
        "revision",
        "invoice_revision",
    ],

    "Tax_Expected": [
        "tax_expected",
        "expected_tax",
        "expected_gst",
        "gst_expected",
        "expected_vat",
        "expected_tax_amount",
    ],

    "Tax_Billed": [
        "tax_billed",
        "tax_charged",
        "tax_amount",
        "gst",
        "gst_amount",
        "gst_charged",
        "gst_billed",
        "vat",
        "vat_amount",
        "tax_actual",
    ],

    "Discount_Applied": [
        "discount_applied",
        "discount",
        "discount_percent",
        "discount_percentage",
        "applied_discount",
        "discount_rate",
    ],

    "Discount_Amount": [
        "discount_amount",
        "discount_value",
        "discount_total",
        "discount_applied_amount",
    ],

    "Billed_Amount": [
        "billed_amount",
        "invoice_total",
        "invoice_amount",
        "invoice_value",
        "bill_amount",
        "bill_total",
        "total_bill",
        "amount_billed",
        "amount_charged",
        "charged_amount",
        "total_amount",
        "total_invoice",
        "net_amount",
        "final_amount",
        "grand_total",
    ],
}


# ============================================================
# NORMALIZE COLUMN NAME
# ============================================================

def normalize_column_name(name):

    name = str(name).strip().lower()

    # Replace special characters with spaces
    name = re.sub(
        r"[^a-z0-9]+",
        "_",
        name
    )

    # Remove duplicate underscores
    name = re.sub(
        r"_+",
        "_",
        name
    )

    # Remove leading/trailing underscores
    name = name.strip("_")

    return name


# ============================================================
# CREATE NORMALIZED ALIAS LOOKUP
# ============================================================

def build_alias_lookup():

    lookup = {}

    for canonical, aliases in COLUMN_ALIASES.items():

        lookup[
            normalize_column_name(canonical)
        ] = canonical

        for alias in aliases:

            lookup[
                normalize_column_name(alias)
            ] = canonical

    return lookup


ALIAS_LOOKUP = build_alias_lookup()


# ============================================================
# FUZZY SIMILARITY
# ============================================================

def similarity(a, b):

    return SequenceMatcher(
        None,
        a,
        b
    ).ratio()


# ============================================================
# MAP COLUMNS
# ============================================================

def map_columns(
    columns,
    fuzzy_threshold=0.88
):

    mapping = {}

    confidence = {}

    used_canonical_fields = set()

    normalized_aliases = list(
        ALIAS_LOOKUP.keys()
    )

    for original_column in columns:

        normalized = normalize_column_name(
            original_column
        )

        # ----------------------------------------------------
        # Exact alias match
        # ----------------------------------------------------

        if normalized in ALIAS_LOOKUP:

            canonical = ALIAS_LOOKUP[
                normalized
            ]

            if canonical not in used_canonical_fields:

                mapping[
                    original_column
                ] = canonical

                confidence[
                    original_column
                ] = 1.0

                used_canonical_fields.add(
                    canonical
                )

            continue

        # ----------------------------------------------------
        # Fuzzy match
        # ----------------------------------------------------

        best_match = None
        best_score = 0

        for alias in normalized_aliases:

            score = similarity(
                normalized,
                alias
            )

            if score > best_score:

                best_score = score
                best_match = alias

        if (
            best_match is not None
            and best_score >= fuzzy_threshold
        ):

            canonical = ALIAS_LOOKUP[
                best_match
            ]

            if canonical not in used_canonical_fields:

                mapping[
                    original_column
                ] = canonical

                confidence[
                    original_column
                ] = round(
                    best_score,
                    4
                )

                used_canonical_fields.add(
                    canonical
                )

    return mapping, confidence


# ============================================================
# APPLY COLUMN MAPPING
# ============================================================

def apply_column_mapping(
    df,
    fuzzy_threshold=0.88
):

    mapping, confidence = map_columns(
        df.columns,
        fuzzy_threshold
    )

    mapped_df = df.rename(
        columns=mapping
    )

    return (
        mapped_df,
        mapping,
        confidence
    )


# ============================================================
# GET UNMAPPED COLUMNS
# ============================================================

def get_unmapped_columns(
    columns,
    mapping
):

    return [
        column
        for column in columns
        if column not in mapping
    ]


# ============================================================
# GET MISSING REQUIRED FIELDS
# ============================================================

def get_missing_required_fields(
    mapped_columns
):

    mapped_columns = set(
        mapped_columns
    )

    return [
        field
        for field in REQUIRED_FIELDS
        if field not in mapped_columns
    ]


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_columns = [

        "invoice_number",
        "customer",
        "actual_consumption",
        "quantity_billed",
        "expected_revenue",
        "invoice_total",
        "gst_expected",
        "gst_charged",
        "discount_percent",
        "unit_rate",

    ]

    mapping, confidence = map_columns(
        test_columns
    )

    print("=" * 70)
    print("COLUMN MAPPING TEST")
    print("=" * 70)

    print("\nMappings:\n")

    for original, canonical in mapping.items():

        print(
            f"{original:30} -> "
            f"{canonical:25} "
            f"confidence={confidence[original]}"
        )

    missing = get_missing_required_fields(
        mapping.values()
    )

    print("\nMissing required fields:")

    if missing:

        for field in missing:
            print(
                f"  ❌ {field}"
            )

    else:

        print(
            "  ✅ All required fields found"
        )