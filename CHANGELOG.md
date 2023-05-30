# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.0.3]
### Changed
- complement type hints
- add python package structure
- add verbose
- make scoring more efficient


## [0.0.4]
### Changed
- add local model explainer module
- reset_index dataframe in discretize() fct in case input data was shuffeled, e.g. from train_test_split. Maight case troubles in explainer 

## [0.0.5]
### Bug fix in explainer
- correct get_explanation() method of Explainer class (line 153)
- df_orig had different column order than nz_freq. This lead to the expaliner assigning the wrong values to the variable names in expalnation column (output of Explainer)
- correct mask in get_feature_names_out() method of onehot_encoder class in utils.py. String search did not work correctly and lead to wrong indexing in pmf's of explainer.py -> make string 
- change logical negation from '~' to 'not' in _make_explanation_string() of explainer.py. 'if ~any(comp):' leads to unexpected outcome.
- Capitalize discretize class
- add global model explanations


## [0.0.7]
### Change threshold logic in explainer
- once most relevant features are determined for local explanations, compute univariate ECDFs for each continuous feature (based on org. scales). 
Then compute the empirical (1-p)% confidence interval of the observations. 
If an observation is not an element of that interval consider it as relevant (w.r.t. anomaly score expl.)     
- Readme.md change 

## [0.0.9]
### explainer module: Change maximum number of bins logic
- If user does not specify a maximum number of bins in the explainer, use a square root of sampe size as a default rule

## [0.1.0]
### Change one_hot_encoder for speed improvements
- Use vectorization in transform method of one_hot_encoder of utils.py. Yields favorable run time improvement