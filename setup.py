from setuptools import setup, find_packages

setup(
    name="data-mining-project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "streamlit",
        "plotly",
        "statsmodels",
    ],
) 