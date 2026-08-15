import os
import joblib
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

from ml.preprocessing.preprocess import prepare_data


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "ml/model.pkl"
THRESHOLD_PATH = "ml/threshold.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            "model.pkl not found. "
            "Run train.py first."
        )

    model = joblib.load(
        MODEL_PATH
    )

    return model


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

def evaluate_thresholds():

    print("\n")
    print("=" * 70)
    print(" REVENUE LEAKAGE - THRESHOLD TUNING")
    print("=" * 70)


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()


    # --------------------------------------------------------
    # Prepare test data
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = prepare_data()


    # --------------------------------------------------------
    # Get probability of Leakage
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X_test
    )[:, 1]


    # --------------------------------------------------------
    # ROC-AUC does not depend on threshold
    # --------------------------------------------------------

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    print(
        f"\nROC-AUC: {roc_auc:.4f}"
    )


    # ========================================================
    # TEST MULTIPLE THRESHOLDS
    # ========================================================

    thresholds = np.arange(
        0.20,
        0.61,
        0.05
    )


    results = []


    print("\n")
    print(
        f"{'Threshold':<12}"
        f"{'Accuracy':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
    )

    print("-" * 60)


    for threshold in thresholds:

        y_pred = (
            probabilities >= threshold
        ).astype(int)


        accuracy = accuracy_score(
            y_test,
            y_pred
        )


        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )


        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )


        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )


        results.append({

            "threshold": threshold,

            "accuracy": accuracy,

            "precision": precision,

            "recall": recall,

            "f1": f1

        })


        print(
            f"{threshold:<12.2f}"
            f"{accuracy:<12.4f}"
            f"{precision:<12.4f}"
            f"{recall:<12.4f}"
            f"{f1:<12.4f}"
        )


    # ========================================================
    # FIND BEST F1 THRESHOLD
    # ========================================================

    best_result = max(
        results,
        key=lambda x: x["f1"]
    )


    best_threshold = (
        best_result["threshold"]
    )


    print("\n")
    print("=" * 70)
    print(" BEST THRESHOLD")
    print("=" * 70)


    print(
        f"\nThreshold : "
        f"{best_threshold:.2f}"
    )


    print(
        f"Accuracy  : "
        f"{best_result['accuracy']:.4f}"
    )


    print(
        f"Precision : "
        f"{best_result['precision']:.4f}"
    )


    print(
        f"Recall    : "
        f"{best_result['recall']:.4f}"
    )


    print(
        f"F1 Score  : "
        f"{best_result['f1']:.4f}"
    )


    # ========================================================
    # FINAL PREDICTIONS
    # ========================================================

    y_pred_best = (
        probabilities >= best_threshold
    ).astype(int)


    print("\n")
    print("=" * 70)
    print(" CLASSIFICATION REPORT")
    print("=" * 70)


    print(
        classification_report(
            y_test,
            y_pred_best,
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
        y_pred_best
    )


    print("\n")
    print("=" * 70)
    print(" CONFUSION MATRIX")
    print("=" * 70)


    print(matrix)


    # ========================================================
    # SAVE THRESHOLD
    # ========================================================

    joblib.dump(
        best_threshold,
        THRESHOLD_PATH
    )


    print("\n")
    print(
        f"Best threshold saved to: "
        f"{THRESHOLD_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    evaluate_thresholds()