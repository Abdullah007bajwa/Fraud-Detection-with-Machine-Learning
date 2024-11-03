# Fraud Detection with Machine Learning

This repository contains a comprehensive machine learning pipeline for detecting fraud in financial transactions. The project uses a combination of preprocessing, resampling techniques, and a Random Forest classifier to handle class imbalance and optimize fraud detection.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Pipeline Overview](#pipeline-overview)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Project Overview
This project addresses fraud detection by:
1. Merging data from multiple sources.
2. Preprocessing features for machine learning, including handling categorical variables and scaling numeric data.
3. Using SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance.
4. Building and training a Random Forest model for classification.
5. Evaluating model performance using metrics like accuracy, precision, recall, F1 score, and a confusion matrix.

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/fraud-detection-pipeline.git
   cd fraud-detection-pipeline
   ```
2. **Install dependencies**:
   Install the required packages via `pip`:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download Dataset**:
   The dataset is automatically downloaded via the Kaggle API. Ensure you have your Kaggle API key configured in the environment.

## Usage
1. **Open in Google Colab**: Use this Jupyter notebook for the full pipeline and visualization.
2. **Run Notebook Cells in Order**:
   - Cell `[14]` downloads the dataset.
   - Cells `[33-35]` load and merge datasets.
   - Cells `[36-48]` preprocess, balance, train, and evaluate the model.

### Example Notebook Commands
The following steps are taken throughout the notebook:

- Load and merge datasets:
    ```python
    session_data = pd.read_csv('session_related.csv')
    outlier_data = pd.read_csv('generic_outliers_data.csv')
    delivery_data = pd.read_csv('delivery_related.csv')
    merged_data = session_data.merge(delivery_data, on='OrderId').merge(outlier_data, on='CustId')
    ```

- Preprocess data and resample using SMOTE:
    ```python
    from imblearn.over_sampling import SMOTE
    X_train_resampled, y_train_resampled = SMOTE(random_state=42).fit_resample(X_train_transformed, y_train)
    ```

- Train model:
    ```python
    model_pipeline.fit(X_train_resampled, y_train_resampled)
    ```

## Dependencies
- `pandas`
- `numpy`
- `sklearn`
- `imblearn`
- `matplotlib`
- `seaborn`

## Pipeline Overview
1. **Data Loading and Merging**: Merges three CSV files containing session, outlier, and delivery data on shared identifiers.
2. **Data Cleaning and Transformation**: Fills missing values and maps categorical fields to binary/one-hot-encoded formats.
3. **Feature Engineering and Preprocessing**: Splits data into categorical and numerical columns for separate preprocessing with scaling and one-hot encoding.
4. **Class Balancing with SMOTE**: Uses SMOTE to balance fraud and non-fraud samples in the training set.
5. **Model Training**: Trains a Random Forest model with the processed data.
6. **Evaluation**: Outputs accuracy, precision, recall, and F1 score, alongside a confusion matrix for model evaluation.

## Results
Sample results for the model on test data:
- **Accuracy**: 0.735
- **Precision**: 0.5 (note: low due to class imbalance)
- **Recall**: 0.71
- **F1 Score**: 0.85
- **Confusion Matrix**: Visualized in the final step of the notebook.

## Acknowledgments
This project leverages data from [Kaggle's fraud detection dataset](https://www.kaggle.com/datasets/kartikkkc/fraud-data).
