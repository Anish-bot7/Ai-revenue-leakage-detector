import numpy as np


def generate_usage(contract):

    actual_usage = np.random.randint(500,3000)

    billed_usage = actual_usage

    unit_price = contract["Unit_Price"]

    expected_amount = actual_usage * unit_price

    return {

        "Billing_Month": np.random.randint(1,13),

        "Billing_Year": 2026,

        "Actual_Usage": actual_usage,

        "Billed_Usage": billed_usage,

        "Peak_Usage": actual_usage + np.random.randint(0,300),

        "OffPeak_Usage": max(0, actual_usage - np.random.randint(0,200)),

        "Expected_Amount": expected_amount,

        "Contract_Price": expected_amount

    }