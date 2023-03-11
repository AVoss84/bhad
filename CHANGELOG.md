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
