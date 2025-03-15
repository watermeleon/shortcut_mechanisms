# Content of setup.py
from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f.readlines()]

setup(
    name="robin_nlp",
    version="0.1.0",
    packages=find_packages(),
    description="Code for paper Robustness and Interpretability of NLP models",
    install_requires=requirements,
)
