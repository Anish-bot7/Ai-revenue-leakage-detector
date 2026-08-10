import os
import pandas as pd
import numpy as np
from faker import Faker

from generators.customer import generate_customer
from generators.contract import generate_contract
from generators.usage import generate_usage
from generators.invoice import generate_invoice
from generators.leakage import inject_leakage

# -----------------------------------------
# Initialize
# -----------------------------------------

fake = Faker()

np.random.seed(42)
Faker.seed(42)

NUM_CUSTOMERS = 500
NUM_CONTRACTS = 200
NUM_INVOICES = 10000

os.makedirs("datasets", exist_ok=True)

# -----------------------------------------
# Generate Customer Master
# -----------------------------------------

customers = []

for i in range(NUM_CUSTOMERS):
    customers.append(generate_customer(i + 1))

customer_df = pd.DataFrame(customers)

# -----------------------------------------
# Generate Contract Master
# -----------------------------------------

contracts = []

for i in range(NUM_CONTRACTS):
    contracts.append(generate_contract(i + 1))

contract_df = pd.DataFrame(contracts)

# -----------------------------------------
# Save Master Data
# -----------------------------------------

customer_df.to_csv(
    "datasets/customers.csv",
    index=False
)

contract_df.to_csv(
    "datasets/contracts.csv",
    index=False
)

# -----------------------------------------
# Generate Invoice Dataset
# -----------------------------------------

records = []

for i in range(NUM_INVOICES):

    # Pick Existing Customer
    customer = customer_df.sample(1).iloc[0].to_dict()

    # Pick Existing Contract
    contract = contract_df.sample(1).iloc[0].to_dict()

    # Generate Usage
    usage = generate_usage(contract)

    # Generate Invoice
    invoice = generate_invoice(contract, usage)

    # Inject Leakage
    invoice = inject_leakage(
        invoice,
        contract,
        usage
    )

    row = {

        "Invoice_ID": f"INV{i+1:06}"

    }

    row.update(customer)
    row.update(contract)
    row.update(usage)
    row.update(invoice)

    records.append(row)

# -----------------------------------------
# Final Dataset
# -----------------------------------------

df = pd.DataFrame(records)

# -----------------------------------------
# Save
# -----------------------------------------

df.to_csv(
    "datasets/revenue_leakage_dataset.csv",
    index=False
)

df.to_excel(
    "datasets/revenue_leakage_dataset.xlsx",
    index=False
)

print("\n==============================")
print(" DATASET GENERATED ")
print("==============================")

print(f"Customers : {len(customer_df)}")
print(f"Contracts : {len(contract_df)}")
print(f"Invoices  : {len(df)}")

print("\nLeakage Distribution")

print(df["Leakage"].value_counts())

print("\nLeakage Types")

print(df["Leakage_Type"].value_counts())

print("\nFiles Created")

print("datasets/customers.csv")
print("datasets/contracts.csv")
print("datasets/revenue_leakage_dataset.csv")
print("datasets/revenue_leakage_dataset.xlsx")

print("\n========== DATASET SUMMARY ==========\n")

print("Shape :", df.shape)

print("\nLeakage Distribution")
print(df["Leakage"].value_counts())

print("\nLeakage Types")
print(df["Leakage_Type"].value_counts())

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows")
print(df.duplicated().sum())