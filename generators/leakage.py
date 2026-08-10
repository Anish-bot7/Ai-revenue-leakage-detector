import numpy as np


def inject_leakage(invoice, contract, usage):

    invoice = invoice.copy()

    # -----------------------------
    # Feature Flags
    # -----------------------------

    invoice["Missing_GST"] = 0
    invoice["Usage_Mismatch"] = 0
    invoice["Extra_Discount"] = 0
    invoice["Wrong_Pricing"] = 0
    invoice["Missing_Charge"] = 0
    invoice["Duplicate_Invoice"] = 0
    invoice["Contract_Violation"] = 0
    invoice["Late_Billing"] = 0

    invoice["Leakage"] = 0
    invoice["Leakage_Type"] = "None"
    invoice["Leakage_Amount"] = 0

    # ---------------------------------
    # 70% Correct Invoice
    # ---------------------------------

    if np.random.rand() < 0.70:
        return invoice

    # ---------------------------------
    # 30% Leakage Invoice
    # ---------------------------------

    leakage_case = np.random.choice([
        "Missing GST",
        "Usage Mismatch",
        "Extra Discount",
        "Wrong Pricing",
        "Missing Charge",
        "Duplicate Invoice",
        "Contract Violation",
        "Late Billing"
    ])

    leakage_amount = 0

    # ==========================================
    # 1 Missing GST
    # ==========================================

    if leakage_case == "Missing GST":

        reduction = invoice["Tax_Billed"] * np.random.uniform(0.10,0.40)

        invoice["Tax_Billed"] -= reduction

        invoice["Billed_Amount"] -= reduction

        leakage_amount = reduction

        invoice["Missing_GST"] = 1


    # ==========================================
    # 2 Usage Mismatch
    # ==========================================

    elif leakage_case == "Usage Mismatch":

        missing_units = np.random.randint(50,250)

        invoice["Billed_Usage"] = max(
            0,
            invoice["Billed_Usage"]-missing_units
        )

        reduction = missing_units * contract["Unit_Price"]

        invoice["Billed_Amount"] -= reduction

        leakage_amount = reduction

        invoice["Usage_Mismatch"] = 1


    # ==========================================
    # 3 Extra Discount
    # ==========================================

    elif leakage_case == "Extra Discount":

        extra = np.random.choice([5,10,15])

        invoice["Discount_Applied"] += extra

        reduction = usage["Expected_Amount"] * extra /100

        invoice["Discount_Amount"] += reduction

        invoice["Billed_Amount"] -= reduction

        leakage_amount = reduction

        invoice["Extra_Discount"] = 1


    # ==========================================
    # 4 Wrong Pricing
    # ==========================================

    elif leakage_case == "Wrong Pricing":

        wrong_price = contract["Unit_Price"] * np.random.uniform(0.7,0.9)

        reduction = (
            contract["Unit_Price"]-wrong_price
        ) * usage["Actual_Usage"]

        invoice["Billed_Amount"] -= reduction

        leakage_amount = reduction

        invoice["Wrong_Pricing"] = 1


    # ==========================================
    # 5 Missing Charge
    # ==========================================

    elif leakage_case == "Missing Charge":

        reduction = usage["Expected_Amount"] * np.random.uniform(0.10,0.30)

        invoice["Billed_Amount"] -= reduction

        leakage_amount = reduction

        invoice["Missing_Charge"] = 1


    # ==========================================
    # 6 Duplicate Invoice
    # ==========================================

    elif leakage_case == "Duplicate Invoice":

        leakage_amount = invoice["Billed_Amount"]

        invoice["Duplicate_Invoice"] = 1


    # ==========================================
    # 7 Contract Violation
    # ==========================================

    elif leakage_case == "Contract Violation":

        reduction = contract["Contract_Price"] * np.random.uniform(0.05,0.20)

        invoice["Billed_Amount"] -= reduction

        leakage_amount = reduction

        invoice["Contract_Violation"] = 1


    # ==========================================
    # 8 Late Billing
    # ==========================================

    elif leakage_case == "Late Billing":

        invoice["Late_Billing"] = 1

        leakage_amount = np.random.randint(500,5000)

    # ---------------------------------

    invoice["Leakage"] = 1

    invoice["Leakage_Type"] = leakage_case

    invoice["Leakage_Amount"] = round(leakage_amount,2)

    invoice["Billed_Amount"] = round(
        max(invoice["Billed_Amount"],0),
        2
    )

    return invoice