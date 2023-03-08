# Bayesian Histogram-based Anomaly Detection (BHAD)

Python code for the BHAD algorithm as presented in [Vosseler, A. (2021): BHAD: Fast unsupervised anomaly detection using Bayesian histograms, Technical Report](https://www.researchgate.net/publication/364265660_BHAD_Fast_unsupervised_anomaly_detection_using_Bayesian_histograms). 

The code follows a standard Scikit-learn API. Code to run the BHAD model is contained in *bhad.py* and some utility functions are provided in *utils.py*, e.g. a discretization function in the case of continuous features as outlined in the references above. 

## Package installation

Create conda virtual environment with required packages 
```bash
conda env create -f env.yml
conda activate env_bhad
pip install -e src
```

## Usage

```python
from bhad.model import BHAD

bm = BHAD(contamination = 0.01, nbins = None)  
```

For a given dataset:

```python
y_pred = bm.fit_predict(X = dataset)        
scores = bm.anomaly_scores             # obtain anomaly scores
```
