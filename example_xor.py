from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


from missing_vals.utils import set_seed, augment_with_missing_values
from missing_vals.compass_net import COMPASSNet
from missing_vals.xor import generate_xor
from missing_vals.model import MissingEstimator


if __name__ == "__main__":
    # NOTE: Add this 
    data_augmentation = True # Set to True to enable data augmentation with missing values
    print("Starting XOR benchmark...")
    for imputer_name in ["custom", "mean", "knn", "iterative", "promissing", "mpromissing"]:
        print(f"Using imputer: {imputer_name}")

        # Initialize lists to store results
        train_acc_list = []
        test_acc_list = []
        train_auc_list = []
        test_auc_list = []
        custom_model = None
        if imputer_name == "custom":
            custom_model = COMPASSNet(in_features=2)
            data_augmentation = True

        print("Training and evaluating the MissingEstimator on the XOR dataset...")
        # for the sake of checking i'll mask 30% of x1
        for random_state in tqdm(range(5)):
            set_seed(random_state)
            datasets = generate_xor()
            nan_indexes = datasets["x1"].sample(frac=0.3, random_state=random_state).index
            datasets.loc[nan_indexes, "x1"] = np.nan
            assert datasets["x1"].isnull().sum() > 0

            training_fraction = 0.5
            train_df = datasets.sample(frac=training_fraction, random_state=42)
            test_df = datasets.drop(train_df.index)
            if data_augmentation:
                train_df = augment_with_missing_values(
                    train_df,
                    augmentation_fraction=0.3,
                    exclude_columns=["target"],
                    random_state=random_state,
                )
                for feature in ["x1", "x2"]:
                    train_df[feature] = train_df[feature].apply(
                        lambda x: x + np.random.normal(0, 0.1) if pd.notnull(x) else x
                    )

            model = MissingEstimator(imputer_name=imputer_name, custom_model=custom_model, epochs=1000, early_stopping=0.1, patience=20, random_state=random_state, output_activation="sigmoid")

            # Fit the model
            model.fit(train_df[["x1", "x2"]].values, train_df["target"].values)

            # Evaluate the model
            train_predictions = model.predict(train_df[["x1", "x2"]].values)
            test_predictions = model.predict(test_df[["x1", "x2"]].values)

            train_accuracy = (train_predictions == train_df["target"].values).mean()
            test_accuracy = (test_predictions == test_df["target"].values).mean()

            train_auc = roc_auc_score(
                train_df["target"].values,
                model.predict_proba(train_df[["x1", "x2"]].values)[:, 1],
            )
            test_auc = roc_auc_score(
                test_df["target"].values,
                model.predict_proba(test_df[["x1", "x2"]].values)[:, 1],
            )

            train_acc_list.append(train_accuracy)
            test_acc_list.append(test_accuracy)
            train_auc_list.append(train_auc)
            test_auc_list.append(test_auc)

        print(
            f"Average Train Accuracy: {np.mean(train_acc_list):.4f} ± {np.std(train_acc_list):.4f}"
        )
        print(
            f"Average Test Accuracy: {np.mean(test_acc_list):.4f} ± {np.std(test_acc_list):.4f}"
        )
        print(
            f"Average Train AUC: {np.mean(train_auc_list):.4f} ± {np.std(train_auc_list):.4f}"
        )
        print(
            f"Average Test AUC: {np.mean(test_auc_list):.4f} ± {np.std(test_auc_list):.4f}"
        )