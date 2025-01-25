import numpy as np
import bhad.utils as utils
from bhad.utils import paste


def test_exp_normalize():
    """Test Exp-normalize trick"""
    x = np.random.normal(size=(1000, 1))
    assert (
        np.round(utils.exp_normalize(x).sum(), 5) == 1
    ), "Input vector does not sum to 1!"


def test_log_post_pmf_nof_bins():
    """Test log posterior prob. measure of number of bins"""
    disc = utils.Discretize(nbins=None, prior_max_M=100, verbose=False)
    x = np.random.normal(size=1000)
    lpost = disc.log_post_pmf_nof_bins(x)
    for p in lpost.values():
        assert np.isfinite(p), f"log post. {p} is nan!"


def test_paste_basic():
    assert paste(["a", "b"], ["c", "d"]) == ["a c", "b d"]


def test_paste_with_separator():
    assert paste(["a", "b"], ["c", "d"], sep="-") == ["a-c", "b-d"]


def test_paste_with_collapse():
    assert paste(["a", "b"], ["c", "d"], collapse=",") == "a c,b d"


def test_paste_with_separator_and_collapse():
    assert paste(["a", "b"], ["c", "d"], sep="-", collapse=",") == "a-c,b-d"


def test_paste_different_lengths():
    assert paste(["a", "b"], ["c"]) == ["a c"]


def test_paste_empty_lists():
    assert paste([], []) == []


def test_paste_single_list():
    assert paste(["a", "b"]) == ["a", "b"]


def test_paste_multiple_lists():
    assert paste(["a", "b"], ["c", "d"], ["e", "f"]) == ["a c e", "b d f"]


# pytest -v tests
