# Bayesian Histogram-based Anomaly Detection (BHAD)

Python implementation of the BHAD algorithm as presented in [Vosseler, A. (2022): Unsupervised Insurance Fraud Prediction Based on Anomaly Detector Ensembles](https://www.researchgate.net/publication/361463552_Unsupervised_Insurance_Fraud_Prediction_Based_on_Anomaly_Detector_Ensembles) and [Vosseler, A. (2023): BHAD: Explainable anomaly detection using Bayesian histograms](https://www.researchgate.net/publication/364265660_BHAD_Explainable_anomaly_detection_using_Bayesian_histograms). The package has been presented at *PyCon DE & PyData Berlin 2023*, you can watch the presentation [here](https://www.youtube.com/watch?v=_8zfgPTD-d8&list=PLGVZCDnMOq0peDguAzds7kVmBr8avp46K&index=8) as well as at *42nd International Workshop on Bayesian Inference and Maximum Entropy Methods in Science and Engineering* ([MaxEnt 2023](https://www.mdpi.com/2673-9984/9/1/1)). The ***bhad* package** follows Scikit-learn's standard API for [outlier detection](https://scikit-learn.org/stable/modules/outlier_detection.html).

## Installation

```bash
pip install bhad
```

## Usage

1.) Preprocess the input data: discretize continuous features and conduct Bayesian model selection (optionally).

2.) Train the model using discrete data.

For convenience these two steps can be wrapped up via a scikit-learn pipeline (optional). 

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
