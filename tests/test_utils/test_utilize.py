import numpy as np
import bhad.utils as utils

def test_exp_normalize():
    """Test Exp-normalize trick
    """
    x = np.random.normal(size=(1000,1))
    assert np.round(utils.exp_normalize(x).sum(),5) == 1, 'Input vector does not sum to 1!'



def test_log_post_pmf_nof_bins():
    """Test log posterior prob. measure of number of bins
    """
    disc = utils.Discretize(nbins = None, prior_max_M = 100, verbose = False)
    x = np.random.normal(size=1000)
    lpost = disc.log_post_pmf_nof_bins(x)
    for p in lpost.values(): 
        assert np.isfinite(p), f"log post. {p} is nan!"


# pytest -v tests