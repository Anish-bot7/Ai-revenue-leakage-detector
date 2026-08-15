"""
ROBUST FLEXIBLE V3 - HYBRID REVENUE LEAKAGE PREDICTION

Flow:

CSV / Excel
    ↓
Column Mapping
    ↓
Normalization
    ↓
Validation
    ↓
Feature Engineering
    ↓
Date Feature Engineering
    ↓
V3 XGBoost
    ↓
ML Anomaly Score
    ↓
Business Evidence
    ↓
Hybrid Decision
    ↓
Prediction + Risk + Potential Leakage + Reasons
    ↓
CSV Result
"""

import os
import joblib
import numpy as np
import pandas as pd

from ml.preprocessing.pipeline import process_uploaded_file


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_PATH = (
    "ml/models/robust_v3_model.pkl"
)

PREPROCESSOR_PATH = (
    "ml/models/robust_v3_preprocessor.pkl"
)

THRESHOLD_PATH = (
    "ml/models/robust_v3_threshold.pkl"
)

FEATURES_PATH = (
    "ml/models/robust_v3_features.pkl"
)


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

def load_artifacts():

    print("\nLoading V3 model...")

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not os.path.exists(
        PREPROCESSOR_PATH
    ):

        raise FileNotFoundError(
            f"Preprocessor not found: "
            f"{PREPROCESSOR_PATH}"
        )

    if not os.path.exists(
        THRESHOLD_PATH
    ):

        raise FileNotFoundError(
            f"Threshold not found: "
            f"{THRESHOLD_PATH}"
        )

    if not os.path.exists(
        FEATURES_PATH
    ):

        raise FileNotFoundError(
            f"Feature definition not found: "
            f"{FEATURES_PATH}"
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    model = joblib.load(
        MODEL_PATH
    )

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    threshold = joblib.load(
        THRESHOLD_PATH
    )

    feature_info = joblib.load(
        FEATURES_PATH
    )

    print(
        "V3 model loaded successfully."
    )

    print(
        f"Threshold: {threshold:.2f}"
    )

    return (
        model,
        preprocessor,
        threshold,
        feature_info
    )


# ============================================================
# SAFE NUMBER CONVERSION
# ============================================================

def safe_number(value):

    try:

        value = float(value)

        if np.isnan(value):

            return None

        return value

    except:

        return None


# ============================================================
# ADD MISSING INDICATORS
# ============================================================

def add_missing_indicators(
    df,
    base_features
):

    df = df.copy()

    for feature in base_features:

        indicator_name = (
            f"{feature}_is_missing"
        )

        df[indicator_name] = (
            df[feature]
            .isna()
            .astype(int)
        )

    return df


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

def prepare_model_input(
    df,
    feature_info
):

    # --------------------------------------------------------
    # Feature definitions saved during training
    # --------------------------------------------------------

    base_features = (
        feature_info[
            "base_features"
        ]
    )

    numeric_features = (
        feature_info[
            "numeric_features"
        ]
    )

    categorical_features = (
        feature_info[
            "categorical_features"
        ]
    )

    indicator_features = (
        feature_info[
            "indicator_features"
        ]
    )

    model_numeric_features = (
        feature_info[
            "model_numeric_features"
        ]
    )

    model_categorical_features = (
        feature_info[
            "model_categorical_features"
        ]
    )

    # --------------------------------------------------------
    # Copy
    # --------------------------------------------------------

    X = df.copy()

    # --------------------------------------------------------
    # Make missing base features
    # --------------------------------------------------------

    for feature in base_features:

        if feature not in X.columns:

            X[feature] = np.nan

    # --------------------------------------------------------
    # Select only base features
    # --------------------------------------------------------

    X = X[
        base_features
    ].copy()

    # --------------------------------------------------------
    # Numerical conversion
    # --------------------------------------------------------

    for feature in numeric_features:

        if feature in X.columns:

            X[feature] = pd.to_numeric(
                X[feature],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Categorical conversion
    # --------------------------------------------------------

    for feature in categorical_features:

        if feature in X.columns:

            X[feature] = (
                X[feature]
                .astype(str)
                .replace(
                    [
                        "",
                        "nan",
                        "None",
                        "NaN"
                    ],
                    np.nan
                )
            )

    # --------------------------------------------------------
    # Missing indicators
    # --------------------------------------------------------

    X = add_missing_indicators(
        X,
        base_features
    )

    # --------------------------------------------------------
    # Ensure all indicator features exist
    # --------------------------------------------------------

    for feature in indicator_features:

        if feature not in X.columns:

            X[feature] = 1

    # --------------------------------------------------------
    # Final column order
    # --------------------------------------------------------

    X = X[
        model_numeric_features
        + model_categorical_features
    ]

    return X


# ============================================================
# BUSINESS EVIDENCE ENGINE
# ============================================================

def calculate_business_evidence(
    row
):

    business_score = 0

    factors = []

    # --------------------------------------------------------
    # 1. EXPECTED VS BILLED AMOUNT
    # --------------------------------------------------------

    expected = safe_number(
        row.get(
            "Expected_Amount"
        )
    )

    billed = safe_number(
        row.get(
            "Billed_Amount"
        )
    )

    amount_difference = None

    if (
        expected is not None
        and billed is not None
    ):

        amount_difference = (
            expected - billed
        )

        # ----------------------------------------------------
        # Billed lower than expected
        # ----------------------------------------------------

        if amount_difference > 0:

            business_score += 50

            factors.append(
                f"Billed amount is "
                f"₹{amount_difference:,.2f} "
                f"lower than expected"
            )

        # ----------------------------------------------------
        # Billed equals expected
        # ----------------------------------------------------

        elif amount_difference == 0:

            factors.append(
                "Billed amount matches "
                "expected amount"
            )

        # ----------------------------------------------------
        # Billed higher than expected
        # ----------------------------------------------------

        else:

            factors.append(
                "Billed amount exceeds "
                "expected amount"
            )

    # ========================================================
    # 2. USAGE DIFFERENCE
    # ========================================================

    actual_usage = safe_number(
        row.get(
            "Actual_Usage"
        )
    )

    billed_usage = safe_number(
        row.get(
            "Billed_Usage"
        )
    )

    usage_difference = None

    if (
        actual_usage is not None
        and billed_usage is not None
    ):

        usage_difference = (
            actual_usage
            - billed_usage
        )

        if (
            abs(usage_difference)
            > 0
        ):

            business_score += 20

            factors.append(
                f"Usage mismatch detected "
                f"({usage_difference:,.2f})"
            )

    # ========================================================
    # 3. TAX DIFFERENCE
    # ========================================================

    tax_expected = safe_number(
        row.get(
            "Tax_Expected"
        )
    )

    tax_billed = safe_number(
        row.get(
            "Tax_Billed"
        )
    )

    tax_difference = None

    if (
        tax_expected is not None
        and tax_billed is not None
    ):

        tax_difference = (
            tax_expected
            - tax_billed
        )

        if tax_difference > 0:

            business_score += 15

            factors.append(
                f"Tax underbilling detected "
                f"(₹{tax_difference:,.2f})"
            )

    # ========================================================
    # 4. DISCOUNT VIOLATION
    # ========================================================

    discount_allowed = safe_number(
        row.get(
            "Discount_Allowed"
        )
    )

    discount_applied = safe_number(
        row.get(
            "Discount_Applied"
        )
    )

    if (
        discount_allowed is not None
        and discount_applied is not None
    ):

        if (
            discount_applied
            > discount_allowed
        ):

            business_score += 15

            factors.append(
                "Applied discount exceeds "
                "allowed discount"
            )

    # ========================================================
    # FINAL SCORE
    # ========================================================

    business_score = min(
        business_score,
        100
    )

    return {

        "business_score":
            business_score,

        "amount_difference":
            amount_difference,

        "usage_difference":
            usage_difference,

        "tax_difference":
            tax_difference,

        "factors":
            factors,

    }


# ============================================================
# HYBRID DECISION ENGINE
# ============================================================

def make_hybrid_decision(
    row,
    ml_probability,
    threshold
):

    evidence = (
        calculate_business_evidence(
            row
        )
    )

    business_score = (
        evidence[
            "business_score"
        ]
    )

    factors = list(
        evidence[
            "factors"
        ]
    )

    # --------------------------------------------------------
    # ML anomaly score
    # --------------------------------------------------------

    ml_score = (
        float(ml_probability)
        * 100
    )

    # ========================================================
    # CASE 1
    # NO BUSINESS EVIDENCE
    # ========================================================

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

    # ========================================================
    # CASE 2
    # STRONG BUSINESS EVIDENCE
    # ========================================================

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

            factors.append(
                "Business rules indicate "
                "potential leakage, but "
                "ML confidence is lower"
            )

    # ========================================================
    # CASE 3
    # MODERATE BUSINESS EVIDENCE
    # ========================================================

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

    # ========================================================
    # CASE 4
    # WEAK BUSINESS EVIDENCE
    # ========================================================

    else:

        prediction = (
            "Potential Anomaly - Review"
        )

        risk_level = "LOW"

    # ========================================================
    # POTENTIAL LEAKAGE
    # ========================================================

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

    # ========================================================
    # FINAL RESULT
    # ========================================================

    return {

        "Prediction":
            prediction,

        "ML_Anomaly_Score":
            round(
                ml_score,
                2
            ),

        "Business_Evidence_Score":
            business_score,

        "Risk_Level":
            risk_level,

        "Potential_Leakage":
            round(
                potential_leakage,
                2
            ),

        "Risk_Factors":
            "; ".join(
                factors
            ),

    }


# ============================================================
# GENERATE RESULTS
# ============================================================

def generate_business_results(
    original_df,
    probabilities,
    threshold
):

    results = []

    for index, probability in enumerate(
        probabilities
    ):

        row = original_df.iloc[
            index
        ]

        # ----------------------------------------------------
        # Invoice ID
        # ----------------------------------------------------

        invoice_id = row.get(
            "Invoice_ID",
            f"ROW_{index + 1}"
        )

        # ----------------------------------------------------
        # Expected / Billed
        # ----------------------------------------------------

        expected = safe_number(
            row.get(
                "Expected_Amount"
            )
        )

        billed = safe_number(
            row.get(
                "Billed_Amount"
            )
        )

        # ----------------------------------------------------
        # Hybrid decision
        # ----------------------------------------------------

        decision = make_hybrid_decision(
            row,
            probability,
            threshold
        )

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        results.append({

            "Invoice_ID":
                invoice_id,

            "Prediction":
                decision[
                    "Prediction"
                ],

            "ML_Anomaly_Score":
                decision[
                    "ML_Anomaly_Score"
                ],

            "Business_Evidence_Score":
                decision[
                    "Business_Evidence_Score"
                ],

            "Risk_Level":
                decision[
                    "Risk_Level"
                ],

            "Expected_Amount":
                expected,

            "Billed_Amount":
                billed,

            "Potential_Leakage":
                decision[
                    "Potential_Leakage"
                ],

            "Risk_Factors":
                decision[
                    "Risk_Factors"
                ],

        })

    return pd.DataFrame(
        results
    )


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def predict_file(
    file_path
):

    print("\n")
    print("=" * 70)
    print(" ROBUST FLEXIBLE V3 HYBRID PREDICTION")
    print("=" * 70)

    # ========================================================
    # STEP 1
    # ========================================================

    print(
        "\n[1/4] Processing uploaded file..."
    )

    pipeline_result = (
        process_uploaded_file(
            file_path
        )
    )

    # --------------------------------------------------------
    # Validation failed
    # --------------------------------------------------------

    if not pipeline_result[
        "success"
    ]:

        print(
            "\n❌ Uploaded file failed "
            "validation."
        )

        return None

    # --------------------------------------------------------
    # Prepared dataframe
    # --------------------------------------------------------

    df = pipeline_result[
        "data"
    ].copy()

    print(
        f"\nPrepared input shape: "
        f"{df.shape}"
    )

    # ========================================================
    # STEP 2
    # ========================================================

    print(
        "\n[2/4] Loading V3 ML model..."
    )

    (
        model,
        preprocessor,
        threshold,
        feature_info
    ) = load_artifacts()

    # ========================================================
    # STEP 3
    # ========================================================

    print(
        "\n[3/4] Preparing model input..."
    )

    X = prepare_model_input(
        df,
        feature_info
    )

    print(
        "Model input shape:",
        X.shape
    )

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    X_processed = (
        preprocessor.transform(
            X
        )
    )

    print(
        "Processed shape:",
        X_processed.shape
    )

    # --------------------------------------------------------
    # ML prediction
    # --------------------------------------------------------

    probabilities = (
        model.predict_proba(
            X_processed
        )[:, 1]
    )

    # ========================================================
    # STEP 4
    # ========================================================

    print(
        "\n[4/4] Combining ML + business evidence..."
    )

    results = generate_business_results(
        df,
        probabilities,
        threshold
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print("\n")
    print("=" * 70)
    print(" HYBRID PREDICTION RESULTS")
    print("=" * 70)

    print(
        results.to_string(
            index=False
        )
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print(" RESULT SUMMARY")
    print("=" * 70)

    print(
        "\nTotal invoices:",
        len(results)
    )

    print(
        "\nPrediction distribution:"
    )

    print(
        results[
            "Prediction"
        ].value_counts()
    )

    print(
        "\nRisk distribution:"
    )

    print(
        results[
            "Risk_Level"
        ].value_counts()
    )

    total_leakage = (
        results[
            "Potential_Leakage"
        ]
        .fillna(0)
        .sum()
    )

    print(
        "\nTotal potential leakage:",
        f"₹{total_leakage:,.2f}"
    )

    # ========================================================
    # SAVE
    # ========================================================

    os.makedirs(
        "results",
        exist_ok=True
    )

    output_path = (
        "results/"
        "robust_v3_hybrid_predictions.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print(
        "\nResults saved to:"
    )

    print(
        output_path
    )

    return results


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_file = (
        "sample_inputs/company_messy.csv"
    )

    predict_file(test_file)

    try:

        predict_file(
            test_file
        )

    except FileNotFoundError as e:

        print(
            "\n❌ File error:"
        )

        print(e)

    except Exception as e:

        print(
            "\n❌ Prediction error:"
        )

        print(
            type(e).__name__,
            ":",
            e
        )