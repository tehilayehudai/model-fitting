import numpy as np
from typing import Any
import json
import click


def get_hyperparameters_grid() -> dict[str, list[Any]]:
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=100, stop=2000, num=20)]
    # Number of features to consider at every split
    max_features = ["auto", "sqrt"]
    # Maximum number of levels in tree
    max_depth: list[int | None] = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(
        None
    )  # The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4, 8]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {
        "n_estimators": n_estimators,
        "max_features": max_features,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "bootstrap": bootstrap,
    }

    return random_grid


def sample_configurations(
    grid: dict[str, list[Any]], sample_size: int, seed: int
) -> list[dict[str, Any]]:
    configurations = []
    conversion_fn = {
        "n_estimators": int,
        "max_features": str,
        "max_depth": lambda x: x,
        "min_samples_split": int,
        "min_samples_leaf": int,
        "bootstrap": bool,
    }
    for i in range(sample_size):
        configuration = {}
        for key in grid:
            np.random.seed(seed + i)
            configuration[key] = conversion_fn[key](
                np.random.choice(grid[key], size=1)[0]
            )
        configurations.append(configuration)
    return configurations


@click.command()
@click.option("--sample_size", type=int, default=100)
@click.option("--seed", type=int, default=42)
def generate_hyperparameters(sample_size: int, seed: int) -> None:
    grid = get_hyperparameters_grid()
    configurations = sample_configurations(grid, sample_size, seed)
    print(json.dumps(configurations, sort_keys=True, indent=4))


if __name__ == "__main__":
    generate_hyperparameters()
