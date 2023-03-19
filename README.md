# Bayesian Histogram-based Anomaly Detection (BHAD)

Python code for the BHAD algorithm as presented in [Vosseler, A. (2021): BHAD: Fast unsupervised anomaly detection using Bayesian histograms, Technical Report](https://www.researchgate.net/publication/364265660_BHAD_Fast_unsupervised_anomaly_detection_using_Bayesian_histograms). 

The code follows a standard Scikit-learn API. Code to run the BHAD model is contained in *bhad.py* and some utility functions are provided in *utils.py*, e.g. a discretization function in the case of continuous features and the Bayesian model selection approach as outlined in the reference. The *explainer.py* module contains code to create individual model explanations. 

## Package installation

Create conda virtual environment with required packages 
```bash
conda env create -f env.yml
conda activate env_bhad
pip install -e src           # install package 
```

## Usage

```python
from sklearn.pipeline import Pipeline
from bhad.utils import Discretize
from bhad.model import BHAD
from bhad.explainer import Explainer

num_cols = [....]
cat_cols = [....]

pipe = Pipeline(steps=[
    ('discrete', Discretize(nbins = None)),   # discretize continous features + model selection
    ('model', BHAD(contamination = 0.01, numeric_features = num_cols, cat_features = cat_cols))
])
```

For a given dataset:

```python
y_pred = pipe.fit_predict(X = dataset)        
```

Get local model explanations, i.e. for each observation:

```python
local_expl = Explainer(pipe.named_steps['model'], pipe.named_steps['discrete']).fit()

df_train = local_expl.get_explanation(nof_feat_expl = 3)

print(local_expl.global_feat_imp)         # List feat. in asc. order of rel. importance
```

