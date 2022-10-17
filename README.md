# Bayesian Histogram-based Anomaly Detection (BHAD)

Python code for the BHAD algorithm as presented in [Vosseler, A. (2022): Unsupervised Insurance Fraud Prediction Based on Anomaly Detector Ensembles, Risks, 10(7), 132](https://www.mdpi.com/2227-9091/10/7/132) and [Vosseler, A. (2021): BHAD: Fast unsupervised anomaly detection using Bayesian histograms, Technical Report](https://www.researchgate.net/publication/364265660_BHAD_Fast_unsupervised_anomaly_detection_using_Bayesian_histograms). 

The code follows a standard Scikit-learn API. Code to run the BHAD model is contained in *bhad.py* and some utility functions are provided in *utils.py*, e.g. a discretization function in the case of continuous features as outlined in the references above. 



```python
import bhad, utils
from ... import dataset                       # your training data
from sklearn.pipeline import Pipeline

pipe = Pipeline(steps=[
    ('discrete' , utils.discretize(nbins = 30)),   # discretizing continous features
    ('model', bhad.BHAD(contamination = 0.01))     
])
```

For a given training dataset:

```python
y_pred = pipe.fit_predict(X = dataset)        # fit + predict
scores = pipe.decision_function(X = dataset)  # obtain anomaly scores
```
