"""
Robust Revenue Leakage Prediction Engine

Flow:

CSV / Excel
    ↓
Input Pipeline
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
Robust XGBoost
    ↓
Leakage Probability
    ↓
Risk Level
    ↓
Potential Leakage
    ↓
Results CSV
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
    "ml/models/robust_flexible_model.pkl"
)

PREPROCESSOR_PATH = (
    "ml/models/robust_flexible_preprocessor.pkl"
)

THRESHOLD_PATH = (
    "ml/models/robust_flexible_threshold.pkl"
)

FEATURES_PATH = (
    "ml/models/robust_flexible_features.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_robust_model():

    required_files = [

        MODEL_PATH,
        PREPROCESSOR_PATH,
        THRESHOLD_PATH,
        FEATURES_PATH

    ]

    for path in required_files:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Required model file not found: {path}"
            )

    model = joblib.load(
        MODEL_PATH
    )

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    threshold = joblib.load(
        THRESHOLD_PATH
    )

    features = joblib.load(
        FEATURES_PATH
    )

    return (
        model,
        preprocessor,
        threshold,
        features
    )


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(
    df,
    feature_list
):

    df = df.copy()

    # --------------------------------------------------------
    # Add missing features
    # --------------------------------------------------------

    for feature in feature_list:

        if feature not in df.columns:

            df[feature] = np.nan

    # --------------------------------------------------------
    # Exact feature order
    # --------------------------------------------------------

    df = df[
        feature_list
    ].copy()

    return df


# ============================================================
# POTENTIAL LEAKAGE
# ============================================================

def calculate_potential_leakage(
    df
):

    if (
        "Expected_Amount" not in df.columns
        or
        "Billed_Amount" not in df.columns
    ):

        return pd.Series(
            np.nan,
            index=df.index
        )

    expected = pd.to_numeric(
        df["Expected_Amount"],
        errors="coerce"
    )

    billed = pd.to_numeric(
        df["Billed_Amount"],
        errors="coerce"
    )

    return (
        expected - billed
    ).clip(
        lower=0
    )


# ============================================================
# RISK LEVEL
# ============================================================

def calculate_risk(
    probability,
    threshold
):

    if probability >= 0.80:

        return "HIGH"

    if probability >= threshold:

        return "MEDIUM"

    return "LOW"


# ============================================================
# RISK FACTORS
# ============================================================

def generate_risk_factors(
    row,
    probability,
    threshold
):

    factors = []

    # --------------------------------------------------------
    # Amount difference
    # --------------------------------------------------------

    if (
        "Expected_Amount" in row
        and
        "Billed_Amount" in row
    ):

        expected = pd.to_numeric(
            pd.Series(
                [row["Expected_Amount"]]
            ),
            errors="coerce"
        ).iloc[0]

        billed = pd.to_numeric(
            pd.Series(
                [row["Billed_Amount"]]
            ),
            errors="coerce"
        ).iloc[0]

        if (
            pd.notna(expected)
            and
            pd.notna(billed)
            and
            expected > billed
        ):

            difference = (
                expected - billed
            )

            factors.append(
                f"Billed amount is "
                f"₹{difference:,.2f} "
                f"lower than expected"
            )

    # --------------------------------------------------------
    # Usage difference
    # --------------------------------------------------------

    if (
        "Usage_Difference" in row
    ):

        value = pd.to_numeric(
            pd.Series(
                [row["Usage_Difference"]]
            ),
            errors="coerce"
        ).iloc[0]

        if (
            pd.notna(value)
            and
            value > 0
        ):

            factors.append(
                f"Usage difference detected: "
                f"{value:,.2f}"
            )

    # --------------------------------------------------------
    # Tax difference
    # --------------------------------------------------------

    if (
        "Tax_Difference" in row
    ):

        value = pd.to_numeric(
            pd.Series(
                [row["Tax_Difference"]]
            ),
            errors="coerce"
        ).iloc[0]

        if (
            pd.notna(value)
            and
            value > 0
        ):

            factors.append(
                f"Tax difference detected: "
                f"{value:,.2f}"
            )

    # --------------------------------------------------------
    # Discount
    # --------------------------------------------------------

    if (
        "Discount_Exceeded" in row
    ):

        value = pd.to_numeric(
            pd.Series(
                [row["Discount_Exceeded"]]
            ),
            errors="coerce"
        ).iloc[0]

        if (
            pd.notna(value)
            and
            value > 0
        ):

            factors.append(
                "Discount exceeds allowed amount"
            )

    # --------------------------------------------------------
    # Generic ML anomaly
    # --------------------------------------------------------

    if not factors:

        if probability >= threshold:

            factors.append(
                "ML model detected a billing anomaly"
            )

        else:

            factors.append(
                "No major billing discrepancy detected"
            )

    return factors


# ============================================================
# PREDICT FILE
# ============================================================

def predict_file(
    file_path
):

    print("\n")
    print("=" * 70)
    print(" ROBUST REVENUE LEAKAGE PREDICTION")
    print("=" * 70)

    # ========================================================
    # STEP 1
    # ========================================================

    print("\n[1/4] Processing uploaded file...")

    pipeline_result = process_uploaded_file(
        file_path
    )

    if not pipeline_result["success"]:

        print(
            "\n❌ File cannot be analyzed."
        )

        return pipeline_result

    df = pipeline_result[
        "data"
    ].copy()

    # ========================================================
    # STEP 2
    # ========================================================

    print(
        "\n[2/4] Loading robust model..."
    )

    (
        model,
        preprocessor,
        threshold,
        feature_list
    ) = load_robust_model()

    print(
        "Robust model loaded."
    )

    print(
        f"Threshold: {threshold:.2f}"
    )

    print(
        f"Expected features: "
        f"{len(feature_list)}"
    )

    # ========================================================
    # STEP 3
    # ========================================================

    print(
        "\n[3/4] Preparing model input..."
    )

    X = prepare_features(
        df,
        feature_list
    )

    print(
        f"Model input shape: {X.shape}"
    )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    X_processed = (
        preprocessor.transform(
            X
        )
    )

    print(
        f"Processed shape: "
        f"{X_processed.shape}"
    )

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    probabilities = (
        model.predict_proba(
            X_processed
        )[:, 1]
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions = (
        probabilities >= threshold
    ).astype(int)

    # ========================================================
    # STEP 4
    # ========================================================

    print(
        "\n[4/4] Generating business results..."
    )

    results = pd.DataFrame()

    # --------------------------------------------------------
    # Invoice ID
    # --------------------------------------------------------

    if "Invoice_ID" in df.columns:

        results[
            "Invoice_ID"
        ] = df[
            "Invoice_ID"
        ].values

    else:

        results[
            "Invoice_ID"
        ] = [
            f"ROW_{i+1:05d}"

            for i in range(
                len(df)
            )
        ]

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    results[
        "Prediction"
    ] = np.where(

        predictions == 1,

        "Revenue Leakage Detected",

        "No Revenue Leakage"

    )

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    results[
        "Leakage_Probability"
    ] = np.round(
        probabilities,
        4
    )

    results[
        "Leakage_Probability_Percentage"
    ] = np.round(
        probabilities * 100,
        2
    )

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    results[
        "Risk_Level"
    ] = [

        calculate_risk(
            probability,
            threshold
        )

        for probability in probabilities

    ]

    # --------------------------------------------------------
    # Expected / billed
    # --------------------------------------------------------

    if "Expected_Amount" in df.columns:

        results[
            "Expected_Amount"
        ] = df[
            "Expected_Amount"
        ].values

    if "Billed_Amount" in df.columns:

        results[
            "Billed_Amount"
        ] = df[
            "Billed_Amount"
        ].values

    # --------------------------------------------------------
    # Potential leakage
    # --------------------------------------------------------

    results[
        "Potential_Leakage"
    ] = calculate_potential_leakage(
        df
    ).values

    # --------------------------------------------------------
    # Risk factors
    # --------------------------------------------------------

    results[
        "Risk_Factors"
    ] = [

        "; ".join(
            generate_risk_factors(
                df.iloc[i],
                probabilities[i],
                threshold
            )
        )

        for i in range(
            len(df)
        )

    ]

    # ========================================================
    # DISPLAY
    # ========================================================

    print("\n")
    print("=" * 70)
    print(" PREDICTION RESULTS")
    print("=" * 70)

    print(
        results.to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    os.makedirs(
        "results",
        exist_ok=True
    )

    output_path = (
        "results/"
        "robust_predictions.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print("\n")
    print(
        "Results saved to:"
    )

    print(
        output_path
    )

    print("\n")
    print("=" * 70)
    print(" PREDICTION COMPLETED")
    print("=" * 70)

    return {

        "success": True,

        "data": results,

        "threshold":
            threshold,

        "mapping":
            pipeline_result[
                "mapping"
            ],

        "confidence":
            pipeline_result[
                "confidence"
            ],

        "validation":
            pipeline_result[
                "validation"
            ]

    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_file = (
        "sample_inputs/company_test.csv"
    )

    try:

        result = predict_file(
            test_file
        )

    except Exception as error:

        print("\n")
        print("=" * 70)
        print(" ❌ PREDICTION ERROR")
        print("=" * 70)

        print(
            f"\n{type(error).__name__}: "
            f"{error}"
        )