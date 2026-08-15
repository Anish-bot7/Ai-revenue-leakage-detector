import os
import joblib
import pandas as pd

from ml.preprocessing.preprocess import create_features, remove_unusable_columns


# ============================================================
# MODEL FILES
# ============================================================

MODEL_PATH = "ml/models/random_forest.pkl"
PREPROCESSOR_PATH = "ml/models/preprocessor.pkl"
THRESHOLD_PATH = "ml/models/threshold.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Random Forest model not found."
        )

    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(
            "Preprocessor not found."
        )

    if not os.path.exists(THRESHOLD_PATH):
        raise FileNotFoundError(
            "Prediction threshold not found."
        )

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    threshold = joblib.load(THRESHOLD_PATH)

    return model, preprocessor, threshold


# ============================================================
# GENERATE EXPLANATION
# ============================================================

def generate_explanation(invoice_data):

    reasons = []

    # --------------------------------------------------------
    # Amount difference
    # --------------------------------------------------------

    expected = float(
        invoice_data.get(
            "Expected_Amount",
            0
        )
    )

    billed = float(
        invoice_data.get(
            "Billed_Amount",
            0
        )
    )

    if billed < expected:

        difference = expected - billed

        reasons.append(
            f"Billed amount is ₹{difference:,.2f} "
            f"lower than expected"
        )

    # --------------------------------------------------------
    # Usage mismatch
    # --------------------------------------------------------

    actual_usage = float(
        invoice_data.get(
            "Actual_Usage",
            0
        )
    )

    billed_usage = float(
        invoice_data.get(
            "Billed_Usage",
            0
        )
    )

    if billed_usage < actual_usage:

        difference = (
            actual_usage - billed_usage
        )

        reasons.append(
            f"Billed usage is {difference:,.0f} "
            f"units lower than actual usage"
        )

    # --------------------------------------------------------
    # Tax discrepancy
    # --------------------------------------------------------

    tax_expected = float(
        invoice_data.get(
            "Tax_Expected",
            0
        )
    )

    tax_billed = float(
        invoice_data.get(
            "Tax_Billed",
            0
        )
    )

    if tax_billed < tax_expected:

        difference = (
            tax_expected - tax_billed
        )

        reasons.append(
            f"Tax billed is ₹{difference:,.2f} "
            f"lower than expected"
        )

    # --------------------------------------------------------
    # Discount discrepancy
    # --------------------------------------------------------

    discount_allowed = float(
        invoice_data.get(
            "Discount_Allowed",
            0
        )
    )

    discount_applied = float(
        invoice_data.get(
            "Discount_Applied",
            0
        )
    )

    if discount_applied > discount_allowed:

        reasons.append(
            f"Applied discount "
            f"({discount_applied}%) exceeds "
            f"allowed discount "
            f"({discount_allowed}%)"
        )

    # --------------------------------------------------------
    # Billing ratio
    # --------------------------------------------------------

    if expected > 0:

        billing_ratio = (
            billed / expected
        )

        if billing_ratio < 0.90:

            reasons.append(
                f"Billing ratio is "
                f"{billing_ratio * 100:.2f}%"
            )

    # --------------------------------------------------------
    # Default
    # --------------------------------------------------------

    if not reasons:

        reasons.append(
            "No major billing discrepancy detected"
        )

    return reasons


# ============================================================
# PREDICT ONE INVOICE
# ============================================================

def predict_invoice(invoice_data):

    # --------------------------------------------------------
    # Convert input into DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(
        [invoice_data]
    )

    # --------------------------------------------------------
    # Feature Engineering
    # --------------------------------------------------------

    X = create_features(df)

    # --------------------------------------------------------
    # Remove columns not used during training
    # --------------------------------------------------------

    X = remove_unusable_columns(
        X
    )

    # --------------------------------------------------------
    # Load trained artifacts
    # --------------------------------------------------------

    model, preprocessor, threshold = (
        load_model()
    )

    # --------------------------------------------------------
    # Apply SAME preprocessor
    # --------------------------------------------------------

    X_processed = preprocessor.transform(
        X
    )

    # --------------------------------------------------------
    # Predict probability
    # --------------------------------------------------------

    probability = model.predict_proba(
        X_processed
    )[0][1]

    # --------------------------------------------------------
    # Apply saved threshold
    # --------------------------------------------------------

    prediction = int(
        probability >= threshold
    )

    # --------------------------------------------------------
    # Business calculation
    # --------------------------------------------------------

    expected_amount = float(
        invoice_data.get(
            "Expected_Amount",
            0
        )
    )

    billed_amount = float(
        invoice_data.get(
            "Billed_Amount",
            0
        )
    )

    potential_leakage = max(
        expected_amount - billed_amount,
        0
    )

    # --------------------------------------------------------
    # Risk level
    # --------------------------------------------------------

    if probability >= 0.80:

        risk_level = "HIGH"

    elif probability >= threshold:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"

    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    reasons = generate_explanation(
        invoice_data
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {

        "prediction": (
            "Revenue Leakage Detected"
            if prediction == 1
            else
            "No Revenue Leakage"
        ),

        "leakage": prediction,

        "probability": round(
            probability,
            4
        ),

        "probability_percentage": round(
            probability * 100,
            2
        ),

        "risk_level": risk_level,

        "expected_amount": round(
            expected_amount,
            2
        ),

        "billed_amount": round(
            billed_amount,
            2
        ),

        "potential_leakage": round(
            potential_leakage,
            2
        ),

        "threshold": round(
            threshold,
            2
        ),

        "risk_factors": reasons
    }

    return result


# ============================================================
# TEST TWO INVOICES
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print(" REVENUE LEAKAGE PREDICTION TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    test_path = (
        "test_data/revenue_leakage_test.csv"
    )

    if not os.path.exists(test_path):

        raise FileNotFoundError(
            f"Test dataset not found: {test_path}"
        )

    df = pd.read_csv(
        test_path,
        keep_default_na=False
    )

    print(
        f"\nTest dataset loaded: {df.shape}"
    )

    # ========================================================
    # NORMAL INVOICE
    # ========================================================

    normal_invoice = df[
        df["Leakage"] == 0
    ].iloc[0].to_dict()

    print("\n")
    print("=" * 70)
    print(" NORMAL INVOICE")
    print("=" * 70)

    normal_result = predict_invoice(
        normal_invoice
    )

    print(
        "\nActual Leakage: 0"
    )

    print(
        "\nPrediction:"
    )

    for key, value in normal_result.items():

        print(
            f"{key}: {value}"
        )

    # ========================================================
    # LEAKAGE INVOICE
    # ========================================================

    leakage_invoice = df[
        df["Leakage"] == 1
    ].iloc[0].to_dict()

    print("\n")
    print("=" * 70)
    print(" LEAKAGE INVOICE")
    print("=" * 70)

    leakage_result = predict_invoice(
        leakage_invoice
    )

    print(
        "\nActual Leakage: 1"
    )

    print(
        "\nPrediction:"
    )

    for key, value in leakage_result.items():

        print(
            f"{key}: {value}"
        )