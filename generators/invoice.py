from datetime import timedelta
from faker import Faker
import numpy as np

fake = Faker()

CURRENCIES = [
    "INR",
    "USD",
    "EUR"
]

PAYMENT_STATUS = [
    "Paid",
    "Pending",
    "Overdue"
]

INVOICE_STATUS = [
    "Generated",
    "Sent",
    "Paid"
]


def generate_invoice(contract, usage):

    invoice_date = fake.date_between(
        start_date="-1y",
        end_date="today"
    )

    due_date = invoice_date + timedelta(days=30)

    tax_expected = usage["Expected_Amount"] * contract["Tax_Rate"] / 100

    tax_billed = tax_expected

    discount_amount = (
        usage["Expected_Amount"]
        * contract["Discount_Allowed"]
        / 100
    )

    discount_applied = contract["Discount_Allowed"]

    billed_amount = (
        usage["Expected_Amount"]
        - discount_amount
        + tax_billed
    )

    return {

        "Invoice_Date": invoice_date,

        "Due_Date": due_date,

        "Currency": np.random.choice(CURRENCIES),

        "Invoice_Status": np.random.choice(INVOICE_STATUS),

        "Payment_Status": np.random.choice(PAYMENT_STATUS),

        "Invoice_Version": np.random.randint(1, 4),

        "Billed_Usage": usage["Actual_Usage"],

        "Tax_Expected": round(tax_expected, 2),

        "Tax_Billed": round(tax_billed, 2),

        "Discount_Applied": discount_applied,

        "Discount_Amount": round(discount_amount, 2),

        "Billed_Amount": round(billed_amount, 2)

    }