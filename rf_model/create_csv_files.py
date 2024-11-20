import pandas as pd
import click
import os.path


def read_num_features(error_rate_path: str) -> list[int]:
    num_features = []
    with open(error_rate_path) as f_input:
        next(f_input)
        for line in f_input:
            num_features.append(int(line.split(sep="\t")[0]))
        return num_features


def read_sorted_features(sorted_features_path: str) -> list[int]:
    sorted_features = []
    with open(sorted_features_path) as f_input:
        for line in f_input:
            sorted_features.append(line.split(sep="\t")[0])
        return sorted_features


@click.command()
@click.argument("hits_path", type=str)
@click.argument("error_rate_path", type=str)
@click.argument("sorted_features_path", type=str)
@click.argument("output_path", type=str)
def create_csv_files(
    hits_path: str, error_rate_path: str, sorted_features_path: str, output_path: str
) -> None:
    hits = pd.read_csv(hits_path)
    num_features = read_num_features(error_rate_path)
    sorted_features = read_sorted_features(sorted_features_path)
    for i in num_features:
        new_df = hits.loc[:, ["sample_name", "label", *sorted_features[:i]]]
        csv_path = os.path.join(output_path, f"Top_{i}_features.csv")
        new_df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    create_csv_files()
