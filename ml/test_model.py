import os
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from preprocess import create_features, remove_unusable_columns


# ============================================================
# CONFIGURATION
# ============================================================

TEST_DATA_PATH = "test_data/revenue_leakage_test.csv"

MODEL_PATH = "ml/model.pkl"

PREPROCESSOR_PATH = "ml/preprocessor.pkl"

THRESHOLD_PATH = "ml/threshold.pkl"


# ============================================================
# LOAD FILES
# ============================================================

def load_artifacts():

    print("\nLoading trained model...")

    model = joblib.load(
        MODEL_PATH
    )

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    threshold = joblib.load(
        THRESHOLD_PATH
    )

    print("Model loaded.")
    print("Preprocessor loaded.")
    print(f"Threshold: {threshold}")

    return (
        model,
        preprocessor,
        threshold
    )


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():

    if not os.path.exists(TEST_DATA_PATH):

        raise FileNotFoundError(
            f"Test dataset not found: "
            f"{TEST_DATA_PATH}"
        )

    df = pd.read_csv(
        TEST_DATA_PATH,
        keep_default_na=False
    )

    print("\nTest dataset loaded.")

    print(
        "Shape:",
        df.shape
    )

    return df


# ============================================================
# PREPARE TEST DATA
# ============================================================

def prepare_test_data(df):

    print("\nPreparing unseen test data...")

    # --------------------------------------------------------
    # Save TRUE target
    # --------------------------------------------------------

    y_test = df["Leakage"].copy()


    # --------------------------------------------------------
    # Feature Engineering
    # --------------------------------------------------------

    X_test = create_features(
        df
    )


    # --------------------------------------------------------
    # Remove columns that were not used by the model
    # --------------------------------------------------------

    X_test = remove_unusable_columns(
        X_test
    )


    X_test = X_test.reset_index(
        drop=True
    )

    y_test = y_test.reset_index(
        drop=True
    )


    return (
        X_test,
        y_test
    )


# ============================================================
# MAIN TEST
# ============================================================

def test_model():

    print("\n")
    print("=" * 70)
    print(" FINAL UNSEEN TEST - REVENUE LEAKAGE DETECTOR")
    print("=" * 70)


    # --------------------------------------------------------
    # Load artifacts
    # --------------------------------------------------------

    (
        model,
        preprocessor,
        threshold
    ) = load_artifacts()


    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    df = load_test_data()


    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    (
        X_test,
        y_test
    ) = prepare_test_data(
        df
    )


    # --------------------------------------------------------
    # Transform using EXISTING preprocessor
    # --------------------------------------------------------

    X_test_processed = preprocessor.transform(
        X_test
    )


    print(
        "\nProcessed test shape:",
        X_test_processed.shape
    )


    # --------------------------------------------------------
    # Predict probabilities
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X_test_processed
    )[:, 1]


    # --------------------------------------------------------
    # Apply saved threshold
    # --------------------------------------------------------

    predictions = (
        probabilities >= threshold
    ).astype(int)


    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print(" FINAL TEST RESULTS")
    print("=" * 70)


    print(
        f"\nAccuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print("\n")
    print("=" * 70)
    print(" CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "No Leakage",
                "Leakage"
            ],
            zero_division=0
        )
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    matrix = confusion_matrix(
        y_test,
        predictions
    )


    print("\n")
    print("=" * 70)
    print(" CONFUSION MATRIX")
    print("=" * 70)

    print(matrix)


    # ========================================================
    # PREDICTION DISTRIBUTION
    # ========================================================

    print("\n")
    print("=" * 70)
    print(" PREDICTION DISTRIBUTION")
    print("=" * 70)

    print(
        pd.Series(
            predictions
        ).value_counts()
    )


    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    results = df.copy()

    results["Predicted_Leakage"] = predictions

    results["Leakage_Probability"] = np.round(
        probabilities,
        4
    )


    results["Prediction_Correct"] = (
        results["Leakage"]
        ==
        results["Predicted_Leakage"]
    )


    results.to_csv(
        "test_data/test_predictions.csv",
        index=False
    )


    results.to_excel(
        "test_data/test_predictions.xlsx",
        index=False
    )


    print("\n")
    print(
        "Prediction files created:"
    )

    print(
        "test_data/test_predictions.csv"
    )

    print(
        "test_data/test_predictions.xlsx"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    test_model()