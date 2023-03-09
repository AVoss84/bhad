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
from sklearn.pipeline import Pipeline
from bhad.utils import discretize
from bhad.model import BHAD

pipe = Pipeline(steps=[
    ('discrete', discretize(nbins = None)),   # discretize continous features + model selection
    ('model', BHAD(contamination = 0.01))     
])
```

For a given dataset:

```python
y_pred = pipe.fit_predict(X = dataset)        
scores = pipe.decision_function(X = dataset)  # obtain anomaly scores
```