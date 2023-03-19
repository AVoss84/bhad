# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [1.0.1]
### Changed
- complement type hints
- add python package structure
- add verbose
- make scoring more efficient


## [1.0.2]
### Changed
- add local model explainer module
- reset_index dataframe in discretize() fct in case input data was shuffeled, e.g. from train_test_split. Maight case troubles in explainer 

## [1.0.3]
### Bug fix in explainer
- correct get_explanation() method of Explainer class (line 153)
- df_orig had different column order than nz_freq. This lead to the expaliner assigning the wrong values to the variable names in expalnation column (output of Explainer)
- correct mask in get_feature_names_out() method of onehot_encoder class in utils.py. String search did not work correctly and lead to wrong indexing in pmf's of explainer.py -> make string 
- change logical negation from '~' to 'not' in _make_explanation_string() of explainer.py. 'if ~any(comp):' leads to unexpected outcome.
- Capitalize discretize class
- add global model explanations


