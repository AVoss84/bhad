from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setup(
    name='bhad',                            # This is the name of the package-distribution (as shown on pypi)
    version='0.0.9',                               # The release version
    author="Alexander Vosseler",                   # Full name of the author
    maintainer = "Alexander Vosseler",
    url='https://github.com/AVoss84/bhad',
    description="Bayesian Histogram-based Anomaly Detection",
    long_description=long_description,             # Long description read from the the readme file
    long_description_content_type="text/markdown",
    keywords = ["bayesian-inference", "anomaly-detection", "unsupervised-learning", "explainability"],
    packages = find_packages(where='src'),         # List of all python modules to be installed
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],                                      # Information to filter the project on PyPi website
    python_requires='>=3.8',                # Minimum version requirement of the package
    py_modules=["bhad"],                    # Name of the python package as used in import
    package_dir={'':'src'},                 # source directory of the main package
    install_requires=['pandas==1.5.*', 'scikit_learn==1.1.*', 'statsmodels==0.13.*', 'matplotlib==3.6.*', 'scipy', 'tqdm'],     # Install other dependencies if any
    extras_require={
        'interactive': ['jupyter', 'ipykernel']
    }
    #,package_data={'exampleproject': ['data/schema.json']}
)

## In case you only use setup.cfg
# if __name__ == "__main__":
#     setup()
