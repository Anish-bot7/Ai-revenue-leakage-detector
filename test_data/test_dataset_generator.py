import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


import os
import pandas as pd
import numpy as np
from faker import Faker

from generators.usage import generate_usage
from generators.invoice import generate_invoice
from generators.leakage import inject_leakage


# ============================================================
# CONFIGURATION
# ============================================================

NUM_TEST_RECORDS = 1000

# Different seed from training dataset
# This is important because we want unseen test data.
np.random.seed(2026)
Faker.seed(2026)

fake = Faker()


# ============================================================
# PROJECT PATHS
# ============================================================

os.makedirs("test_data", exist_ok=True)


# ============================================================
# LOAD EXISTING MASTER DATA
# ============================================================

customers = pd.read_csv(
    "datasets/customers.csv"
)

contracts = pd.read_csv(
    "datasets/contracts.csv"
)


print("\n======================================")
print(" GENERATING TEST DATASET")
print("======================================")

print("Customers available :", len(customers))
print("Contracts available :", len(contracts))
print("Test records        :", NUM_TEST_RECORDS)


# ============================================================
# GENERATE TEST INVOICES
# ============================================================

records = []


for i in range(NUM_TEST_RECORDS):

    # --------------------------------------------------------
    # Select an existing customer
    # --------------------------------------------------------

    customer = (
        customers
        .sample(
            n=1,
            random_state=np.random.randint(0, 1000000)
        )
        .iloc[0]
        .to_dict()
    )


    # --------------------------------------------------------
    # Select an existing contract
    # --------------------------------------------------------

    contract = (
        contracts
        .sample(
            n=1,
            random_state=np.random.randint(0, 1000000)
        )
        .iloc[0]
        .to_dict()
    )


    # --------------------------------------------------------
    # Generate NEW usage
    # --------------------------------------------------------

    usage = generate_usage(contract)


    # --------------------------------------------------------
    # Generate NEW invoice
    # --------------------------------------------------------

    invoice = generate_invoice(
        contract,
        usage
    )


    # --------------------------------------------------------
    # Inject leakage
    #
    # This creates the TRUE answer.
    # The ML model will NOT receive these leakage flags
    # as input during prediction.
    # --------------------------------------------------------

    invoice = inject_leakage(
        invoice,
        contract,
        usage
    )


    # --------------------------------------------------------
    # Create record
    # --------------------------------------------------------

    row = {

        "Invoice_ID":
            f"TEST{i + 1:06}"

    }


    row.update(customer)

    row.update(contract)

    row.update(usage)

    row.update(invoice)


    records.append(row)


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(records)


# ============================================================
# COLUMN ORDER
# ============================================================

column_order = [

    # -------------------------
    # Invoice
    # -------------------------

    "Invoice_ID",


    # -------------------------
    # Customer
    # -------------------------

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


    # -------------------------
    # Contract
    # -------------------------

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


    # -------------------------
    # Usage
    # -------------------------

    "Billing_Month",
    "Billing_Year",
    "Actual_Usage",
    "Billed_Usage",
    "Peak_Usage",
    "OffPeak_Usage",
    "Expected_Amount",


    # -------------------------
    # Invoice
    # -------------------------

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


    # -------------------------
    # Leakage indicators
    # -------------------------

    "Missing_GST",
    "Usage_Mismatch",
    "Extra_Discount",
    "Wrong_Pricing",
    "Missing_Charge",
    "Duplicate_Invoice",
    "Contract_Violation",
    "Late_Billing",


    # -------------------------
    # Ground truth
    # -------------------------

    "Leakage_Type",
    "Leakage_Amount",
    "Leakage"

]


df = df[column_order]


# ============================================================
# SAVE TEST DATASET
# ============================================================

df.to_csv(
    "test_data/revenue_leakage_test.csv",
    index=False
)

df.to_excel(
    "test_data/revenue_leakage_test.xlsx",
    index=False
)


# ============================================================
# TEST DATASET SUMMARY
# ============================================================

print("\n======================================")
print(" TEST DATASET SUMMARY")
print("======================================")

print("\nShape:")
print(df.shape)


print("\nLeakage Distribution:")
print(
    df["Leakage"].value_counts()
)


print("\nLeakage Percentage:")
print(
    df["Leakage"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


print("\nLeakage Types:")
print(
    df["Leakage_Type"].value_counts()
)


print("\nMissing Values:")
print(
    df.isnull().sum().sum()
)


print("\nDuplicate Rows:")
print(
    df.duplicated().sum()
)


print("\n======================================")
print(" TEST DATASET CREATED SUCCESSFULLY")
print("======================================")

print(
    "\nCSV   : test_data/revenue_leakage_test.csv"
)

print(
    "Excel : test_data/revenue_leakage_test.xlsx"
)