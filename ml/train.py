import os
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from preprocess import prepare_data


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "ml/model.pkl"

RANDOM_STATE = 42


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():

    print("\n")
    print("=" * 60)
    print(" REVENUE LEAKAGE DETECTION - MODEL TRAINING")
    print("=" * 60)


    # --------------------------------------------------------
    # Prepare Data
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = prepare_data()


    print("\nData preparation completed.")

    print(
        "\nTraining features:",
        X_train.shape[1]
    )

    print(
        "Training records:",
        X_train.shape[0]
    )

    print(
        "Testing records:",
        X_test.shape[0]
    )


    # --------------------------------------------------------
    # Create Random Forest
    # --------------------------------------------------------

    print("\nCreating Random Forest model...")


    model = RandomForestClassifier(

        n_estimators=300,

        max_depth=None,

        min_samples_split=5,

        min_samples_leaf=2,

        max_features="sqrt",

        class_weight="balanced",

        random_state=RANDOM_STATE,

        n_jobs=-1

    )


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nTraining model...")

    model.fit(
        X_train,
        y_train
    )


    print("Training completed!")


    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )


    y_probability = model.predict_proba(
        X_test
    )[:, 1]


    # ========================================================
    # EVALUATION
    # ========================================================

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

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )


    # --------------------------------------------------------
    # Print Metrics
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print(" MODEL PERFORMANCE")
    print("=" * 60)

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
    # Classification Report
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print(" CLASSIFICATION REPORT")
    print("=" * 60)

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "No Leakage",
                "Leakage"
            ],
            zero_division=0
        )
    )


    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print(" CONFUSION MATRIX")
    print("=" * 60)

    matrix = confusion_matrix(
        y_test,
        y_pred
    )

    print(matrix)

    print("\nFormat:")
    print(
        "[[True Negative   False Positive]"
    )
    print(
        " [False Negative  True Positive]]"
    )


    # ========================================================
    # SAVE MODEL
    # ========================================================

    joblib.dump(
        model,
        MODEL_PATH
    )


    print("\n")
    print("=" * 60)
    print(" MODEL SAVED")
    print("=" * 60)

    print(
        f"\nModel location: {MODEL_PATH}"
    )


    return model


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_model()