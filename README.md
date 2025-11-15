# Heart Disease Prediction Using Machine Learning

**Project summary**

A complete pipeline for predicting heart disease risk using machine learning and bio-inspired optimization algorithms. This project includes data preprocessing, exploratory data analysis (EDA), model training (Random Forest as primary model), hyperparameter tuning and feature selection using bio-inspired algorithms (Bat Algorithm, Bee Algorithm), model evaluation, and a Flask web interface for inference.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Features](#features)
* [Dataset](#dataset)
* [Repository Structure](#repository-structure)
* [Getting Started](#getting-started)

  * [Requirements](#requirements)
  * [Installation](#installation)
* [Usage](#usage)

  * [Training a model](#training-a-model)
  * [Running the Flask app (Web UI / API)](#running-the-flask-app-web-ui--api)
  * [Example API request (curl)](#example-api-request-curl)
* [Modeling Details](#modeling-details)

  * [Preprocessing](#preprocessing)
  * [Models & Optimization](#models--optimization)
  * [Evaluation Metrics](#evaluation-metrics)
* [Results](#results)
* [Saving & Loading Models](#saving--loading-models)
* [Tips & Troubleshooting](#tips--troubleshooting)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)

---

## Project Overview

This project aims to predict whether a patient has heart disease using clinical features (age, gender, blood pressure, cholesterol, ECG results, etc.). It demonstrates an end-to-end ML workflow:

1. Ingest and clean the dataset.
2. Apply feature engineering and selection.
3. Train models (Random Forest as primary baseline).
4. Use Bat Algorithm and Bee Algorithm for bio-inspired optimization to improve feature selection and hyperparameters.
5. Evaluate models with robust metrics and visualize results.
6. Deploy a Flask-based web UI and REST API for inference.

## Features

* Clean, modular code for preprocessing, training, and inference.
* Bio-inspired optimization (Bat Algorithm, Bee Algorithm) for feature selection and hyperparameter search.
* Random Forest baseline and easy hooks to add other models (XGBoost, Logistic Regression, Neural Nets).
* Flask web UI + REST API for quick predictions.
* Model persistence (save/load trained models).
* Example Jupyter notebooks for EDA and experimentation.

## Dataset

This project uses the commonly available heart disease dataset (e.g., UCI Heart Disease / `heart.csv`). Place your dataset at `data/heart.csv` or update the path in the config file.

Columns typically include (but may vary by dataset):

```
age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target
```

* `target` is the label where `1` indicates presence of heart disease and `0` indicates absence.

## Repository Structure

```
heart-disease-prediction/
├─ data/
│  └─ heart.csv
├─ notebooks/
│  └─ eda.ipynb
├─ src/
│  ├─ data_preprocessing.py
│  ├─ feature_selection.py      # implementations for Bat & Bee algorithms
│  ├─ models.py                 # model classes (train, predict wrappers)
│  ├─ train.py                  # training script
│  ├─ evaluate.py               # evaluation utilities
│  └─ predict.py                # CLI prediction
├─ app/
│  ├─ app.py                    # Flask app
│  └─ templates/
│     └─ index.html
├─ models/
│  └─ rf_model.pkl
├─ requirements.txt
├─ README.md
└─ LICENSE
```

## Getting Started

### Requirements

* Python 3.8+
* Recommended packages (example):

```
pandas
numpy
scikit-learn
flask
matplotlib
joblib
scipy
jupyter
```

A full `requirements.txt` is included in the repo.

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
```

2. (Optional) Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate    # macOS / Linux
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Place your dataset in `data/heart.csv`.

## Usage

### Training a model

To train a Random Forest model with default preprocessing and optimization (if enabled):

```bash
python src/train.py --data data/heart.csv --model rf --save models/rf_model.pkl
```

To run training with Bat or Bee optimization for feature selection/hyperparameter tuning, pass flags to `train.py` (examples depend on implementation):

```bash
python src/train.py --data data/heart.csv --model rf --optimize bat --save models/rf_model_bat.pkl
```

The training script will output validation metrics and save the best model to `models/`.

### Running the Flask app (Web UI / API)

Start the Flask app:

```bash
cd app
python app.py
```

By default the app runs on `http://127.0.0.1:5000/`. The web UI lets you enter patient features and get a prediction. The REST API endpoint is described below.

### Example API request (curl)

Predict using the `/predict` endpoint (adjust fields to match your model input):

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}'
```

Sample JSON response:

```json
{ "prediction": 1, "probability": 0.87 }
```

## Modeling Details

### Preprocessing

* Handle missing values (drop or impute depending on missingness).
* Encode categorical features (one-hot or ordinal as appropriate).
* Scale numerical features when required (StandardScaler or MinMaxScaler).
* Optionally perform feature engineering (e.g., interaction terms).

### Models & Optimization

* **Random Forest**: Main baseline model. Robust to feature scaling and outliers.
* **Bat Algorithm**: Bio-inspired optimization used here for feature selection and/or hyperparameter tuning. It mimics echolocation behavior to explore the search space.
* **Bee Algorithm**: Another bio-inspired method used for searching feature subsets or tuning hyperparameters using scout/forager analogy.

Both Bat and Bee algorithm implementations are available in `src/feature_selection.py`. Use them to:

* Select a compact subset of features that preserves (or improves) model performance.
* Perform heuristic hyperparameter search for models (e.g., `n_estimators`, `max_depth`, `min_samples_split`).

### Evaluation Metrics

Common metrics included:

* Accuracy
* Precision, Recall, F1-score
* ROC AUC
* Confusion matrix

The project logs and saves the most important metrics after training to `reports/` (if enabled).

## Results

Include your best model performance here (replace with your actual numbers):

```
Model: Random Forest (with Bat Algorithm feature selection)
Accuracy: 0.86
ROC AUC: 0.92
Precision: 0.84
Recall: 0.88
```

Add confusion matrix and ROC curves in `notebooks/` or `reports/` for visual inspection.

## Saving & Loading Models

Trained models are saved using `joblib` or `pickle` to the `models/` folder. Example:

```python
# save
import joblib
joblib.dump(model, 'models/rf_model.pkl')

# load
model = joblib.load('models/rf_model.pkl')
```

## Tips & Troubleshooting

* If results are poor, check for data leakage, class imbalance, or incorrect preprocessing.
* Use `class_weight='balanced'` or resampling (SMOTE) for imbalanced labels.
* Ensure the Flask app uses the same preprocessing pipeline as the training script.
* Track experiments with simple logs or tools like MLflow/Weights & Biases.

## Contributing

Contributions are welcome. Typical contributions:

* Add new optimization algorithms or models.
* Improve preprocessing and feature engineering.
* Add unit tests and CI configuration.

Please open issues or PRs on the repository.

## License

This project is released under the MIT License. See `LICENSE` for details.

## Contact

Maintainer: `yourname <youremail@example.com>`

Acknowledgements:

* UCI Machine Learning Repository (heart disease dataset)
* Any relevant libraries and references used for Bat/Bee algorithm implementations

---

*Generated README — edit the sections marked with placeholders (dataset path, results, contact info) to match your project specifics.*
