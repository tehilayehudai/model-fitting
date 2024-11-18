import pandas as pd
import numpy as np
import numpy.typing as npt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import operator
import click


def read_hits(path: str) -> tuple[npt.ArrayLike, npt.ArrayLike, list[str]]:
    """Read the table of motif hits across replicates."""
    counts = pd.read_csv(path)
    feature_names = list(counts.iloc[:, 2:].columns)
    X = counts.iloc[:, 2:].to_numpy()
    y = np.where(np.logical_not((counts.iloc[:, 1] == "other").to_numpy()), 1, 0)
    return (X, y, feature_names)


def select_perfect(
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    feature_names: list[str],
    seed: int,
    cv_num_of_splits: int,
) -> list[tuple[str, float]]:

    feature_scores = []
    rf = RandomForestClassifier(random_state=np.random.seed(seed))
    for i, feature in enumerate(feature_names):
        cv_score = cross_val_score(
            rf,
            X[:, i].reshape(-1, 1),
            y,
            cv=StratifiedKFold(cv_num_of_splits, shuffle=True),
        ).mean()
        feature_scores.append((feature, float(cv_score)))
    return feature_scores


def print_scores(feature_scores: list[tuple[str, float]]) -> None:
    print("Feature\tAccuracy_on_cv")
    for feature, score in sorted(
        feature_scores, key=operator.itemgetter(1), reverse=True
    ):
        print(f"{feature}\t{score}")


@click.command()
@click.argument("path")
@click.option("--seed", type=int, default=42)
@click.option("--cv_num_of_splits", type=int, default=2)
def evaluate_single_features(path: str, seed: int, cv_num_of_splits: int) -> None:
    X, y, feature_names = read_hits(path)
    feature_scores = select_perfect(X, y, feature_names, seed, cv_num_of_splits)
    print_scores(feature_scores)


if __name__ == "__main__":
    evaluate_single_features()
