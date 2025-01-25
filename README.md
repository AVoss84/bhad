# 🔥 *Bayesian Histogram Anomaly Detection (BHAD)* 🔥

Python implementation of the *Bayesian Histogram-based Anomaly Detection (BHAD)* algorithm, see [Vosseler, A. (2022): Unsupervised Insurance Fraud Prediction Based on Anomaly Detector Ensembles](https://www.researchgate.net/publication/361463552_Unsupervised_Insurance_Fraud_Prediction_Based_on_Anomaly_Detector_Ensembles) and [Vosseler, A. (2023): BHAD: Explainable anomaly detection using Bayesian histograms](https://www.researchgate.net/publication/364265660_BHAD_Explainable_anomaly_detection_using_Bayesian_histograms). The package was presented at *PyCon DE & PyData Berlin 2023* ([watch talk here](https://www.youtube.com/watch?v=_8zfgPTD-d8&list=PLGVZCDnMOq0peDguAzds7kVmBr8avp46K&index=8)) and at the *42nd International Workshop on Bayesian Inference and Maximum Entropy Methods in Science and Engineering* ([MaxEnt 2023](https://www.mdpi.com/2673-9984/9/1/1)), at Max-Planck-Institute for Plasma Physics, Garching, Germany. 

## Package installation

We opt here for using [*uv*](https://github.com/astral-sh/uv) as a package manager due to its speed and stability, but the same installation works using *pip* with *venv* for Python 3.12: 
```bash
# curl -LsSf https://astral.sh/uv/install.sh | sh          # Optional: install uv for the first time
uv venv env_bhad --python 3.12                             # create the usual virtual environment
source env_bhad/bin/activate
```

For local development (only):
```bash
uv pip install -r pyproject.toml  

# Install bhad in editable mode (incl. notebook dependencies)
uv pip install -e ".[notebook]"
```

Install directly from PyPi:
```bash
pip install bhad                                       
# uv pip install bhad                                     # or via uv
```


## Model usage

1.) Preprocess the input data: discretize continuous features and conduct Bayesian model selection (*optional*).

2.) Train the model using discrete data.

For convenience these two steps can be wrapped up via a scikit-learn pipeline (*optional*). 

```python
from sklearn.pipeline import Pipeline
from bhad.model import BHAD
from bhad.utils import Discretize

num_cols = [....]   # names of numeric features
cat_cols = [....]   # categorical features

# Setting nbins = None infers the Bayes-optimal number of bins (=only parameter)
# using the MAP estimate
pipe = Pipeline(steps=[
   ('discrete', Discretize(nbins = None)),   
   ('model', BHAD(contamination = 0.01, num_features = num_cols, cat_features = cat_cols))
])
```

For a given dataset get binary model decisons and anomaly scores:

```python
y_pred = pipe.fit_predict(X = dataset)        

anomaly_scores = pipe.decision_function(dataset)
```

Get *global* model explanation as well as for *individual* observations:

```python
from bhad.explainer import Explainer

local_expl = Explainer(bhad_obj = pipe.named_steps['model'], discretize_obj = pipe.named_steps['discrete']).fit()

local_expl.get_explanation(nof_feat_expl = 5, append = False)          # individual explanations

print(local_expl.global_feat_imp)                                      # global explanation
```

A detailed *toy example* using synthetic data can be found [here](https://github.com/AVoss84/bhad/blob/main/src/notebooks/Toy_Example.ipynb). An example using the Titanic dataset illustrating *model explanability* with BHAD can be found [here](https://github.com/AVoss84/bhad/blob/main/src/notebooks/Titanic_Example.ipynb).
