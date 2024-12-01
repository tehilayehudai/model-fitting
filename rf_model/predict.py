import click
import joblib
import pandas as pd
import os
import operator


def write_error_rate(error_summary, output_path):
    with open(output_path, "w") as f:
        f.write(f"number_of_features\terror_rate\tnum_of_errors\n")
        for number_of_features, error_rate, error_sum in sorted(error_summary, key=operator.itemgetter(0)):
            f.write(f"{number_of_features}\t{error_rate}\t{error_sum}\n")


def prediction(model_path: str, sample_name, X, y, output_file):
    model = joblib.load(model_path)
    model_score = model.score(X, y)
    error_rate = 1 - model_score
    predictions = model.predict(X)
    errors = predictions != y
    with open(output_file, "w") as f:
        f.write(f"Error rate: {error_rate}\n")
        f.write(f"sample name\ttrue label\tprediction\tprediction score\n")
        for i in range(len(predictions)):
            f.write(f"{sample_name[i]}\t{y[i]}\t{predictions[i]}\t{int(errors[i])}\n")
    return (errors.mean(), errors.sum())


@click.command()
@click.argument("best_model_path", type=str)
@click.argument("hits_path", type=str)
@click.argument("output_path", type=str)
def predict(best_model_path: str, hits_path: str, output_path: str) -> None:
    error_summary = []
    data = pd.read_csv(hits_path)
    os.makedirs(output_path, exist_ok=True)
    if os.path.isdir(best_model_path):
        model_paths = [
            os.path.join(best_model_path, file_name)
            for file_name in os.listdir(best_model_path)
            if file_name.startswith("Top") and file_name.endswith("pkl")
        ]
    for model_path in model_paths:  # pkl files
        number_of_features = int(os.path.split(model_path)[-1].split("_")[1])
        model = joblib.load(model_path)
        feature_names = model.feature_names_in_
        if len(feature_names) != number_of_features:
            raise Exception(f"{len(feature_names)} != {number_of_features}")
        sample_name = data["sample_name"]
        y = data["label"]
        X = data.drop(["label", "sample_name"], axis=1).reindex(feature_names, axis=1)
        output_file = os.path.join(
            output_path, f"Top_{number_of_features}_model_predictions.txt"
        )
        error_summary.append(
            (number_of_features,)
            + prediction(model_path, sample_name, X, y, output_file)
        )
    output_error_summary = os.path.join(output_path, "error_rates.txt")
    write_error_rate(error_summary, output_error_summary)


if __name__ == "__main__":
    predict()
