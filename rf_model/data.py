import pandas as pd
import numpy as np
import numpy.typing as npt


def read_hits(path: str) -> tuple[npt.ArrayLike, npt.ArrayLike, list[str], pd.Series]:
    """Read the table of motif hits across replicates."""
    counts = pd.read_csv(path)
    feature_names = list(counts.iloc[:, 2:].columns)
    sample_names = counts.iloc[:,0]
    # X = counts.iloc[:, 2:].to_numpy()
    # y = np.where(np.logical_not((counts.iloc[:, 1] == "other").to_numpy()), 1, 0)
    X = counts.iloc[:, 2:]
    y = counts['label']
    return (X, y, feature_names, sample_names)
