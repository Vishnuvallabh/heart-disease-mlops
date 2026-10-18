"""Small fake datasets so tests run fast and need no internet."""
import numpy as np
import pandas as pd
import pytest

from preprocess import COLUMNS


@pytest.fixture
def raw_df():
    """6 rows shaped like the raw file, with gaps in ca and thal."""
    rows = [
        [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6, 0],
        [67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, 3, 3, 2],
        [67, 1, 4, 120, 229, 0, 2, 129, 1, 2.6, 2, 2, 7, 1],
        [37, 1, 3, 130, 250, 0, 0, 187, 0, 3.5, 3, np.nan, 3, 0],
        [41, 0, 2, 130, 204, 0, 2, 172, 0, 1.4, 1, 0, np.nan, 4],
        [56, 1, 2, 120, 236, 0, 0, 178, 0, 0.8, 1, 0, 3, 0],
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


@pytest.fixture
def clean_df():
    """100 random but valid-looking patients with both classes present."""
    rng = np.random.default_rng(0)
    n = 100
    return pd.DataFrame({
        "age": rng.integers(29, 78, n),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n),
        "trestbps": rng.integers(94, 200, n),
        "chol": rng.integers(126, 564, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": rng.integers(71, 202, n),
        "exang": rng.integers(0, 2, n),
        "oldpeak": rng.uniform(0, 6.2, n).round(1),
        "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n),
        "thal": rng.choice([3, 6, 7], n),
        "target": np.tile([0, 1], n // 2),
    })
