import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from ml.preprocessing.preprocess import prepare_data


MODEL_PATH = "ml/xgboost_model.pkl"
THRESHOLD_PATH = "ml/xgboost_threshold.pkl"


def tune_xgboost():

    print("\n")
    print("=" * 70)
    print(" XGBOOST THRESHOLD TUNING")
    print("=" * 70)

    # --------------------------------------------------------
    # Prepare validation data
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = prepare_data()

    # --------------------------------------------------------
    # Load XGBoost
    # --------------------------------------------------------

    model = joblib.load(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    print(
        f"\nROC-AUC: {roc_auc:.4f}"
    )

    # --------------------------------------------------------
    # Thresholds
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Best F1
    # --------------------------------------------------------

    best = max(
        results,
        key=lambda x: x["f1"]
    )

    print("\n")
    print("=" * 70)
    print(" BEST XGBOOST THRESHOLD")
    print("=" * 70)

    print(
        f"\nThreshold : {best['threshold']:.2f}"
    )

    print(
        f"Accuracy  : {best['accuracy']:.4f}"
    )

    print(
        f"Precision : {best['precision']:.4f}"
    )

    print(
        f"Recall    : {best['recall']:.4f}"
    )

    print(
        f"F1 Score  : {best['f1']:.4f}"
    )

    # --------------------------------------------------------
    # Save threshold
    # --------------------------------------------------------

    joblib.dump(
        best["threshold"],
        THRESHOLD_PATH
    )

    print(
        f"\nThreshold saved to:"
        f" {THRESHOLD_PATH}"
    )


if __name__ == "__main__":

    tune_xgboost()