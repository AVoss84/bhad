# Bayesian Histogram-based Anomaly Detection (BHAD)

Python implementation of the BHAD algorithm as presented in [Vosseler, A. (2022): Unsupervised Insurance Fraud Prediction Based on Anomaly Detector Ensembles, Risks, 10(7), 132](https://www.mdpi.com/2227-9091/10/7/132) and [Vosseler, A. (2022): BHAD: Fast unsupervised anomaly detection using Bayesian histograms](https://www.researchgate.net/publication/364265660_BHAD_Fast_unsupervised_anomaly_detection_using_Bayesian_histograms). The code follows scikit-learn's standard API for [outlier detection](https://scikit-learn.org/stable/modules/outlier_detection.html). 

## Installation

```bash
pip install bhad
```

## Usage

1.) Preprocess the input data: discretize continuous features and conduct Bayesian model selection (optionally).

2.) Train the model using discrete data.

For convenience these two steps can be wrapped up via a scikit-learn pipeline (optionally). 

```python
from bhad.model import BHAD
from bhad.utils import Discretize
from sklearn.pipeline import Pipeline

num_cols = [....]   # names of numeric features
cat_cols = [....]   # categorical features

pipe = Pipeline(steps=[
   ('discrete', Discretize(nbins = None)),   
   ('model', BHAD(contamination = 0.01, num_features = num_cols, cat_features = cat_cols))
])
```

For a given dataset get binary model decisons:

```python
y_pred = pipe.fit_predict(X = dataset)        
```

Get global model explanation as well as for individual observations:

```python
from bhad.explainer import Explainer

local_expl = Explainer(pipe.named_steps['model'], pipe.named_steps['discrete']).fit()

local_expl.get_explanation(nof_feat_expl = 5, append = False)   # individual explanations

local_expl.global_feat_imp                                      # global explanation
```

A detailed toy example using synthetic data for anomaly detection can be found [here](https://github.com/AVoss84/bhad/blob/main/src/notebooks/Toy_Example.ipynb) and an example using the Titanic dataset illustrating model explanability can be found [here](https://github.com/AVoss84/bhad/blob/main/src/notebooks/Titanic_Example.ipynb).