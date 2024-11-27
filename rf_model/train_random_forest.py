import json
from typing import Any
from rf_model.data import read_hits
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import click
import numpy.typing as npt
from copy import deepcopy
from operator import itemgetter
import joblib
import os
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pandas as pd


def load_configuration(configurations_json, index) -> dict[str, Any]:
    with open(configurations_json, "r") as json_file:
        return json.load(json_file)[index]


def initial_train(X, y, configuration, random_forest_seed):
    rf = RandomForestClassifier(**configuration, random_state=random_forest_seed)
    initial_model = rf.fit(X, y)
    importances = initial_model.feature_importances_
    decreasing_feature_importance = np.argsort(importances)[::-1]
    return (rf, initial_model, importances, decreasing_feature_importance)


def iterative_train(
    rf: RandomForestClassifier,
    model: RandomForestClassifier,
    feature_names,
    decreasing_feature_importance,
    X,
    y: npt.ArrayLike,
    cv_num_of_splits: int,
):
    original_X = X[:]
    _, number_of_features = X.shape
    cv_avg_error_rate = previous_cv_avg_error_rate = 1
    cv_avg_error_rates = []
    number_of_features_per_model = []
    output_models = []
    while cv_avg_error_rate <= previous_cv_avg_error_rate and number_of_features >= 1:

        # save previous cv_avg_error_rate to make sure the performances do not deteriorate
        previous_cv_avg_error_rate = cv_avg_error_rate

        # compute current model accuracy for each fold of the cross validation
        cv_score = cross_val_score(model, X, y, cv=StratifiedKFold(cv_num_of_splits))

        # current model cv_avg_error_rate rate
        cv_avg_error_rate = 1 - cv_score.mean()
        number_of_features_per_model.append(number_of_features)
        cv_avg_error_rates.append(cv_avg_error_rate)

        if cv_avg_error_rate > previous_cv_avg_error_rate:
            break

        # save the model itself (serialized) for future use
        output_models.append(deepcopy(model))

        # update number of features
        number_of_features //= 2

        # extract only the (new) half most important features
        features_indexes_to_keep = decreasing_feature_importance[:number_of_features]
        X = original_X[feature_names[features_indexes_to_keep]]

        if number_of_features > 0:
            model = rf.fit(X, y)

    return cv_avg_error_rates, number_of_features_per_model, output_models


def write_error_rates(output_path, cv_avg_error_rates, number_of_features_per_model):
    file_name = "error_rate.txt"
    output_file = os.path.join(output_path, file_name)
    with open(output_file, "w") as f_handle:
        f_handle.write(f"Features\tErrors\n")
        for num, error in sorted(
            zip(number_of_features_per_model, cv_avg_error_rates), key=itemgetter(0)
        ):
            f_handle.write(f"{num}\t{error}\n")


def write_feature_importances(output_path, importances, feature_names):
    file_name = "sorted_feature_importance.txt"
    output_file = os.path.join(output_path, file_name)
    with open(output_file, "w") as f_handle:
        for feature, impo in sorted(
            zip(feature_names, importances), key=itemgetter(1), reverse=True
        ):
            f_handle.write(f"{feature}\t{impo}\n")


def write_hyperparameters(output_path, configuration):
    file_name = "hyperparameters_configuration.txt"
    output_file = os.path.join(output_path, file_name)
    with open(output_file, "w") as f_handle:
        json.dump(configuration, f_handle, sort_keys=True, indent=4)


def write_model(output_path, number_of_features_per_model, output_models):
    for num, model in zip(number_of_features_per_model, output_models):
        joblib.dump(model, os.path.join(output_path, f"Top_{num}_features_model.pkl"))


## CSV files will print only for the best model


@click.command()
@click.argument("hits_path", type=str)
@click.argument("output_path", type=str)
@click.argument("conf_path", type=str)
@click.argument("conf_index", type=int)
@click.option("--cv_num_of_splits", type=int, default=2)
@click.option("--seed", type=int, default=42)
def train_rf(
    hits_path: str,
    output_path: str,
    conf_path: str,
    conf_index: int,
    cv_num_of_splits: int,
    seed: int,
) -> None:
    X, y, feature_names, sample_names = read_hits(hits_path)
    X_df = pd.DataFrame(X)
    X_df.columns = feature_names
    configuration = load_configuration(conf_path, conf_index)
    rf, initial_model, importances, decreasing_feature_importance = initial_train(
        X_df, y, configuration, seed
    )
    cv_avg_error_rates, number_of_features_per_model, output_models = iterative_train(
        rf,
        initial_model,
        np.array(feature_names),
        decreasing_feature_importance,
        X_df,
        y,
        cv_num_of_splits,
    )
    write_error_rates(output_path, cv_avg_error_rates, number_of_features_per_model)
    write_feature_importances(output_path, importances, feature_names)
    write_hyperparameters(output_path, configuration)
    write_model(output_path, number_of_features_per_model, output_models)


if __name__ == "__main__":
    train_rf()
