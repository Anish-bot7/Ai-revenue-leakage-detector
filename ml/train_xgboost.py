import os
import joblib

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from preprocess import prepare_data


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "ml/xgboost_model.pkl"

RANDOM_STATE = 42


# ============================================================
# TRAIN XGBOOST
# ============================================================

def train_xgboost():

    print("\n")
    print("=" * 70)
    print(" XGBOOST - REVENUE LEAKAGE DETECTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = prepare_data()

    # --------------------------------------------------------
    # Calculate class ratio
    # --------------------------------------------------------

    negative = (y_train == 0).sum()
    positive = (y_train == 1).sum()

    scale_pos_weight = negative / positive

    print("\nClass distribution:")

    print("No Leakage:", negative)
    print("Leakage:", positive)

    print(
        "\nScale Pos Weight:",
        round(scale_pos_weight, 3)
    )

    # --------------------------------------------------------
    # Create XGBoost model
    # --------------------------------------------------------

    model = XGBClassifier(

        n_estimators=400,

        max_depth=6,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        min_child_weight=2,

        gamma=0,

        scale_pos_weight=scale_pos_weight,

        objective="binary:logistic",

        eval_metric="logloss",

        random_state=RANDOM_STATE,

        n_jobs=-1

    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nTraining XGBoost...")

    model.fit(
        X_train,
        y_train
    )

    print("Training completed.")

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(" XGBOOST PERFORMANCE")
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

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\n")
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

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        f"\nXGBoost model saved to:"
        f" {MODEL_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_xgboost()