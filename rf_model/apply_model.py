import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # type: ignore


def read_hits(path: str) -> pd.DataFrame:
    """Read the table of motif hits across replicates."""
    return pd.read_csv(path)


def prepare_data(counts: pd.DataFrame) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    """Get a tuple of the matrix count and the label vector."""
    X = counts.iloc[:, 2:].to_numpy()
    y = np.where(np.logical_not((counts.iloc[:, 1] == "other").to_numpy()), 1, 0)
    return (X, y)


def train_model(X: npt.ArrayLike, y: npt.ArrayLike) -> RandomForestClassifier:
    """Train a random forest model."""
    clf = RandomForestClassifier()
    return clf.fit(X, y)


def predict_model(model: RandomForestClassifier, new_X: npt.ArrayLike) -> npt.ArrayLike:
    """Apply the specified model to a new matrix of motif counts."""
    return model.predict(new_X)


def apply_model(train_path: str, test_path: str, out_path: str) -> None:
    """Train a random forest model and apply it to test data."""
    train_dataset = read_hits(train_path)
    X_train, y_train = prepare_data(train_dataset)
    model = train_model(X_train, y_train)
    test_dataset = read_hits(test_path)
    X_test, y_test = prepare_data(test_dataset)
    y_predicted = predict_model(model, X_test)
    result = pd.DataFrame(
        {
            "Replicate": test_dataset.iloc[:, 0],
            "Original": y_test,
            "Predicted": y_predicted,
            "Score": np.where(y_test == y_predicted, 1, 0),
        }
    )
    result.to_csv(out_path, index=False)
