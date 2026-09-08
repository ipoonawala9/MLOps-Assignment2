# MLOps Assignment 2 — Reproducible ML Pipeline with Git and DVC

## Objective

The objective of this project is to build a reproducible machine learning experiment pipeline using Git and DVC.

The project demonstrates how data, source code, parameters, experiment metrics, and ML pipeline stages can be version-controlled. Different versions of the model parameters are used to perform multiple experiments, and DVC is used to reproduce the results.

The Iris dataset is used as the machine learning dataset.

---

## Technologies Used

- Python 3.11
- pandas
- scikit-learn
- Git
- GitHub
- DVC
- PyYAML
- Random Forest Classifier
- Google Drive — DVC remote storage

---

## Project Structure

```text
MLOps-Assignment2/
│
├── data/
│   ├── dataset.csv.dvc
│   └── .gitignore
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
│
├── models/
│   └── .gitignore
│
├── metrics/
│   └── metrics.json
│
├── params.yaml
├── dvc.yaml
├── dvc.lock
├── requirements.txt
├── .dvcignore
└── README.md