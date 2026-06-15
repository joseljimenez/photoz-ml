import numpy as np

def run_regression(
    regerssor,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
) -> np.ndarray:
    """Run as scikit-learn regression algorithm

    Parameters
    ----------
    regerssor
        Scikit-learn regressor

    train_features:
        Features in training data

    train_targets:
        Targets in training data

    test_features:
        Features in test data

    Returns
    -------
    Predictions on test data
    """
    regerssor.fit(train_features, train_targets)
    return regerssor.predict(test_features)