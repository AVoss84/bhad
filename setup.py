from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='bhad',                     # This is the name of the package
    version='0.0.1',                        # The initial release version
    author="Alexander Vosseler",                     # Full name of the author
    description="Bayesian Histogram-based Anomaly Detection",
    long_description=long_description,      # Long description read from the the readme file
    long_description_content_type="text/markdown",
    packages=find_packages(),    # List of all python modules to be installed
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],                                      # Information to filter the project on PyPi website
    python_requires='>=3.8',                # Minimum version requirement of the package
    py_modules=["bhad"],             # Name of the python package
    package_dir={'files':'src/bhad'},     # Directory of the source code of the package
    install_requires=['pandas==1.5.*', 'scikit_learn==1.1.2', 'statsmodels==0.13.*', 'scipy', 'tqdm'],                     # Install other dependencies if any
    extras_require={
        'interactive': ['matplotlib==3.6.*', 'jupyter', 'ipykernel']
    }
    #,package_data={'exampleproject': ['data/schema.json']}
)

