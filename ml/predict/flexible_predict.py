import os
import joblib
import pandas as pd
import numpy as np

from ml.preprocessing.pipeline import process_uploaded_file


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_PATH = "ml/models/flexible_model.pkl"

PREPROCESSOR_PATH = (
    "ml/models/flexible_preprocessor.pkl"
)

THRESHOLD_PATH = (
    "ml/models/flexible_threshold.pkl"
)

FEATURES_PATH = (
    "ml/models/flexible_features.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_flexible_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(
            f"Preprocessor not found: {PREPROCESSOR_PATH}"
        )

    if not os.path.exists(THRESHOLD_PATH):
        raise FileNotFoundError(
            f"Threshold not found: {THRESHOLD_PATH}"
        )

    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(
            f"Feature list not found: {FEATURES_PATH}"
        )

    model = joblib.load(MODEL_PATH)

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
# PREPARE MODEL INPUT
# ============================================================

def prepare_model_input(
    df,
    features
):

    df = df.copy()

    # --------------------------------------------------------
    # Create missing model features
    # --------------------------------------------------------

    for feature in features:

        if feature not in df.columns:

            df[feature] = np.nan

    # --------------------------------------------------------
    # Keep exact training feature order
    # --------------------------------------------------------

    df = df[
        features
    ].copy()

    return df


# ============================================================
# CALCULATE POTENTIAL LEAKAGE
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

    leakage = (
        expected - billed
    )

    return leakage.clip(
        lower=0
    )


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(
    probability,
    threshold
):

    if probability >= 0.80:

        return "HIGH"

    elif probability >= threshold:

        return "MEDIUM"

    else:

        return "LOW"


# ============================================================
# PREDICT UPLOADED FILE
# ============================================================

def predict_uploaded_file(
    file_path
):

    print("\n")
    print("=" * 70)
    print(" FLEXIBLE REVENUE LEAKAGE PREDICTION")
    print("=" * 70)

    # ========================================================
    # STEP 1
    # ========================================================

    print("\n[1/3] Processing uploaded file...")

    pipeline_result = process_uploaded_file(
        file_path
    )

    # --------------------------------------------------------
    # Stop if validation failed
    # --------------------------------------------------------

    if not pipeline_result["success"]:

        return pipeline_result

    df = pipeline_result["data"]

    # ========================================================
    # STEP 2
    # ========================================================

    print(
        "\n[2/3] Loading flexible ML model..."
    )

    (
        model,
        preprocessor,
        threshold,
        features
    ) = load_flexible_model()

    print(
        "Model loaded successfully."
    )

    print(
        f"Prediction threshold: {threshold:.2f}"
    )

    print(
        f"Expected model features: {len(features)}"
    )

    # ========================================================
    # PREPARE FEATURES
    # ========================================================

    X = prepare_model_input(
        df,
        features
    )

    print(
        f"Model input shape: {X.shape}"
    )

    # ========================================================
    # STEP 3
    # ========================================================

    print(
        "\n[3/3] Running prediction..."
    )

    # --------------------------------------------------------
    # Apply EXACT training preprocessor
    # --------------------------------------------------------

    X_processed = (
        preprocessor.transform(X)
    )

    print(
        f"Processed input shape: "
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
    # CREATE RESULTS
    # ========================================================

    results = df.copy()

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

    results[
        "Leakage_Prediction"
    ] = predictions

    results[
        "Prediction"
    ] = np.where(
        predictions == 1,
        "Revenue Leakage Detected",
        "No Revenue Leakage"
    )

    # ========================================================
    # RISK
    # ========================================================

    results[
        "Risk_Level"
    ] = [

        get_risk_level(
            probability,
            threshold
        )

        for probability in probabilities

    ]

    # ========================================================
    # POTENTIAL LEAKAGE
    # ========================================================

    results[
        "Potential_Leakage"
    ] = calculate_potential_leakage(
        df
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    print("\n")
    print("=" * 70)
    print(" PREDICTION RESULTS")
    print("=" * 70)

    display_columns = [

        "Invoice_ID",

        "Prediction",

        "Leakage_Probability_Percentage",

        "Risk_Level",

        "Expected_Amount",

        "Billed_Amount",

        "Potential_Leakage"

    ]

    # Only display columns that exist

    display_columns = [

        column

        for column in display_columns

        if column in results.columns

    ]

    print(
        results[
            display_columns
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    output_directory = "results"

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    output_path = (
        f"{output_directory}/"
        "flexible_predictions.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print("\n")
    print(
        f"Results saved to:"
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
            ],

        "threshold":
            threshold

    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_file = (
        "test_data/revenue_leakage_test.csv"
    )

    try:

        result = predict_uploaded_file(
            test_file
        )

    except Exception as e:

        print("\n")
        print("=" * 70)
        print(" ❌ PREDICTION ERROR")
        print("=" * 70)

        print(
            f"\n{type(e).__name__}: {e}"
        )