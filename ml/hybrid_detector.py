"""
Hybrid Revenue Leakage Detection Engine

Combines:
    1. Direct business evidence
    2. Usage evidence
    3. Tax evidence
    4. Discount evidence
    5. ML anomaly score

The ML model does NOT blindly determine the final result.
"""

import numpy as np
import pandas as pd


# ============================================================
# SAFE NUMBER
# ============================================================

def number(value):

    try:

        value = float(value)

        if np.isnan(value):
            return None

        return value

    except:

        return None


# ============================================================
# BUSINESS EVIDENCE
# ============================================================

def calculate_business_evidence(row):

    score = 0

    factors = []

    # --------------------------------------------------------
    # Amount difference
    # --------------------------------------------------------

    expected = number(
        row.get("Expected_Amount")
    )

    billed = number(
        row.get("Billed_Amount")
    )

    amount_difference = None

    if (
        expected is not None
        and billed is not None
    ):

        amount_difference = (
            expected - billed
        )

        # Actual direct revenue loss
        if amount_difference > 0:

            score += 50

            factors.append(
                f"Billed amount is "
                f"₹{amount_difference:,.2f} "
                f"lower than expected"
            )

        elif amount_difference < 0:

            factors.append(
                "Billed amount exceeds "
                "expected amount"
            )

    # --------------------------------------------------------
    # Usage difference
    # --------------------------------------------------------

    actual_usage = number(
        row.get("Actual_Usage")
    )

    billed_usage = number(
        row.get("Billed_Usage")
    )

    if (
        actual_usage is not None
        and billed_usage is not None
    ):

        usage_difference = (
            actual_usage
            - billed_usage
        )

        if abs(usage_difference) > 0:

            score += 20

            factors.append(
                f"Usage mismatch detected "
                f"({usage_difference:,.2f})"
            )

    # --------------------------------------------------------
    # Tax difference
    # --------------------------------------------------------

    tax_expected = number(
        row.get("Tax_Expected")
    )

    tax_billed = number(
        row.get("Tax_Billed")
    )

    if (
        tax_expected is not None
        and tax_billed is not None
    ):

        tax_difference = (
            tax_expected
            - tax_billed
        )

        if tax_difference > 0:

            score += 15

            factors.append(
                f"Tax underbilling detected "
                f"(₹{tax_difference:,.2f})"
            )

    # --------------------------------------------------------
    # Discount
    # --------------------------------------------------------

    discount_allowed = number(
        row.get("Discount_Allowed")
    )

    discount_applied = number(
        row.get("Discount_Applied")
    )

    if (
        discount_allowed is not None
        and discount_applied is not None
    ):

        if discount_applied > discount_allowed:

            score += 15

            factors.append(
                "Applied discount exceeds "
                "allowed discount"
            )

    return {

        "business_score":
            min(score, 100),

        "amount_difference":
            amount_difference,

        "factors":
            factors,

    }


# ============================================================
# FINAL DECISION
# ============================================================

def make_hybrid_decision(
    row,
    ml_probability,
    threshold=0.40
):

    evidence = (
        calculate_business_evidence(
            row
        )
    )

    business_score = (
        evidence["business_score"]
    )

    factors = list(
        evidence["factors"]
    )

    # --------------------------------------------------------
    # ML score
    # --------------------------------------------------------

    ml_score = (
        float(ml_probability)
        * 100
    )

    # --------------------------------------------------------
    # CRITICAL RULE
    #
    # If there is NO business evidence,
    # don't call it confirmed leakage.
    # --------------------------------------------------------

    if business_score == 0:

        if ml_probability >= 0.80:

            prediction = (
                "Potential Anomaly - Review"
            )

            risk_level = "MEDIUM"

            factors.append(
                "ML model detected an "
                "anomaly, but no direct "
                "revenue-loss evidence "
                "was found"
            )

        else:

            prediction = (
                "No Revenue Leakage"
            )

            risk_level = "LOW"

            factors.append(
                "No major revenue-loss "
                "evidence detected"
            )

    # --------------------------------------------------------
    # Strong business evidence
    # --------------------------------------------------------

    elif business_score >= 50:

        prediction = (
            "Revenue Leakage Detected"
        )

        if ml_probability >= 0.80:

            risk_level = "HIGH"

        elif ml_probability >= threshold:

            risk_level = "HIGH"

        else:

            risk_level = "MEDIUM"

    # --------------------------------------------------------
    # Moderate evidence
    # --------------------------------------------------------

    elif business_score >= 20:

        prediction = (
            "Potential Revenue Leakage"
        )

        risk_level = "MEDIUM"

        if ml_probability >= 0.70:

            factors.append(
                "ML model supports "
                "the anomaly"
            )

    # --------------------------------------------------------
    # Weak evidence
    # --------------------------------------------------------

    else:

        prediction = (
            "Potential Anomaly - Review"
        )

        risk_level = "LOW"

    # --------------------------------------------------------
    # Potential leakage
    # --------------------------------------------------------

    potential_leakage = 0

    if (
        evidence[
            "amount_difference"
        ] is not None
    ):

        potential_leakage = max(
            evidence[
                "amount_difference"
            ],
            0
        )

    return {

        "Prediction":
            prediction,

        "Risk_Level":
            risk_level,

        "ML_Anomaly_Score":
            round(
                ml_score,
                2
            ),

        "Business_Evidence_Score":
            business_score,

        "Potential_Leakage":
            round(
                potential_leakage,
                2
            ),

        "Risk_Factors":
            "; ".join(factors),

    }