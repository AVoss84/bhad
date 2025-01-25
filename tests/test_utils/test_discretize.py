import pytest
import pandas as pd
import numpy as np
from bhad.utils import Discretize


@pytest.fixture
def sample_data():
    data = {
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
        "feature3": np.random.randn(100),
    }
    return pd.DataFrame(data)


def test_discretize_init():
    discretizer = Discretize(
        columns=["feature1", "feature2"],
        nbins=5,
        lower=0,
        k=2,
        round_intervals=3,
        eps=0.01,
        make_labels=True,
        verbose=False,
    )
    assert discretizer.columns == ["feature1", "feature2"]
    assert discretizer.nof_bins == 5
    assert discretizer.lower == 0
    assert discretizer.k == 2
    assert discretizer.round_intervals == 3
    assert discretizer.eps == 0.01
    assert discretizer.make_labels == True
    assert discretizer.verbose == False


def test_discretize_fit(sample_data):
    discretizer = Discretize(columns=["feature1", "feature2"], nbins=5, verbose=False)
    discretizer.fit(sample_data)
    assert discretizer.nbins == 5
    assert "feature1" in discretizer.save_binnings
    assert "feature2" in discretizer.save_binnings


def test_discretize_transform(sample_data):
    # Only use the columns we want to discretize
    test_data = sample_data[["feature1", "feature2"]]
    discretizer = Discretize(columns=["feature1", "feature2"], nbins=5, verbose=False)
    discretizer.fit(test_data)
    transformed_data = discretizer.transform(test_data)


def test_discretize_fit_transform(sample_data):
    # Only use the columns we want to discretize
    test_data = sample_data[["feature1", "feature2"]]
    discretizer = Discretize(columns=["feature1", "feature2"], nbins=5, verbose=False)
    transformed_data = discretizer.fit_transform(test_data)
