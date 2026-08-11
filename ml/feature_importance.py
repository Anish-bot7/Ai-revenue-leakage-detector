import joblib
import pandas as pd

from preprocess import prepare_data


MODEL_PATH = "ml/model.pkl"


def analyze_feature_importance():

    print("\n")
    print("=" * 70)
    print(" RANDOM FOREST FEATURE IMPORTANCE")
    print("=" * 70)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = joblib.load(
        MODEL_PATH
    )

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
    # Get transformed feature names
    # --------------------------------------------------------

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    # --------------------------------------------------------
    # Get importance
    # --------------------------------------------------------

    importance = model.feature_importances_

    # --------------------------------------------------------
    # Create dataframe
    # --------------------------------------------------------

    importance_df = pd.DataFrame({

        "Feature": feature_names,

        "Importance": importance

    })

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Print top 30
    # --------------------------------------------------------

    print("\nTop 30 Features:\n")

    print(
        importance_df.head(30).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    importance_df.to_csv(
        "ml/feature_importance.csv",
        index=False
    )

    print(
        "\nSaved:"
        " ml/feature_importance.csv"
    )


if __name__ == "__main__":

    analyze_feature_importance()