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
    confusion_matrix
)

from preprocess import create_features, remove_unusable_columns


# ============================================================
# FILE PATHS
# ============================================================

TEST_DATA_PATH = "test_data/revenue_leakage_test.csv"

RF_MODEL_PATH = "ml/model.pkl"
RF_THRESHOLD_PATH = "ml/threshold.pkl"

XGB_MODEL_PATH = "ml/xgboost_model.pkl"
XGB_THRESHOLD_PATH = "ml/xgboost_threshold.pkl"

PREPROCESSOR_PATH = "ml/preprocessor.pkl"


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():

    df = pd.read_csv(
        TEST_DATA_PATH,
        keep_default_na=False
    )

    print("\nTest dataset:")
    print("Shape:", df.shape)

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_test_data(df):

    y_test = df["Leakage"].copy()

    X_test = create_features(df)

    X_test = remove_unusable_columns(
        X_test
    )

    X_test = X_test.reset_index(
        drop=True
    )

    y_test = y_test.reset_index(
        drop=True
    )

    return X_test, y_test


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    name,
    model,
    threshold,
    X_test,
    y_test
):

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

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

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    print("\n")
    print("=" * 60)
    print(name)
    print("=" * 60)

    print(
        f"Threshold : {threshold:.2f}"
    )

    print(
        f"Accuracy  : {accuracy:.4f}"
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

    print("\nConfusion Matrix:")

    print(matrix)

    return {
        "Model": name,
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": roc_auc
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print(" FINAL MODEL COMPARISON")
    print(" COMPLETELY UNSEEN TEST DATA")
    print("=" * 70)

    # --------------------------------------------------------
    # Load preprocessor
    # --------------------------------------------------------

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    random_forest = joblib.load(
        RF_MODEL_PATH
    )

    xgboost = joblib.load(
        XGB_MODEL_PATH
    )

    # --------------------------------------------------------
    # Load thresholds
    # --------------------------------------------------------

    rf_threshold = joblib.load(
        RF_THRESHOLD_PATH
    )

    xgb_threshold = joblib.load(
        XGB_THRESHOLD_PATH
    )

    print(
        "\nRandom Forest threshold:",
        rf_threshold
    )

    print(
        "XGBoost threshold:",
        xgb_threshold
    )

    # --------------------------------------------------------
    # Load unseen test data
    # --------------------------------------------------------

    df = load_test_data()

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    X_test, y_test = prepare_test_data(
        df
    )

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    X_test_processed = preprocessor.transform(
        X_test
    )

    print(
        "\nProcessed test shape:",
        X_test_processed.shape
    )

    # --------------------------------------------------------
    # Evaluate RF
    # --------------------------------------------------------

    rf_result = evaluate_model(
        "RANDOM FOREST",
        random_forest,
        rf_threshold,
        X_test_processed,
        y_test
    )

    # --------------------------------------------------------
    # Evaluate XGBoost
    # --------------------------------------------------------

    xgb_result = evaluate_model(
        "XGBOOST",
        xgboost,
        xgb_threshold,
        X_test_processed,
        y_test
    )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    results = pd.DataFrame([
        rf_result,
        xgb_result
    ])

    print("\n")
    print("=" * 70)
    print(" FINAL COMPARISON")
    print("=" * 70)

    display = results.copy()

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score",
        "ROC_AUC"
    ]

    for metric in metrics:

        display[metric] = (
            display[metric] * 100
        ).round(2)

    print(
        display.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    best_index = results[
        "F1_Score"
    ].idxmax()

    best_model = results.loc[
        best_index,
        "Model"
    ]

    print("\n")
    print("=" * 70)
    print(
        f" FINAL WINNER BASED ON F1: {best_model}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results.to_csv(
        "ml/final_model_comparison.csv",
        index=False
    )

    print(
        "\nSaved:"
        " ml/final_model_comparison.csv"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()