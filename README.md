# Heart_failure_prediction
A machine learning model using logistic regression to predict heart failure mortality based on clinical features, with interactive patient data input and comprehensive visualization of model performance metrics.

## Overview
This project implements a machine learning model to predict heart failure mortality using logistic regression. It provides an interactive interface for medical professionals to input patient data and receive mortality risk predictions.

## Features
- Heart failure mortality prediction using logistic regression
- Interactive patient data input interface
- Visualization of model performance metrics
- Clinical features analysis and importance
- Model evaluation metrics and results

## Prerequisites
- Python 3.7+
- Required packages:
  - scikit-learn
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - streamlit (if using web interface)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Heart_failure_prediction.git
   cd Heart_failure_prediction
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. To train the model:
   ```bash
   python train_model.py
   ```

2. To run the prediction interface:
   ```bash
   python predict.py
   ```
   or if using Streamlit:
   ```bash
   streamlit run app.py
   ```

## Dataset
The model uses clinical features including:
- Age
- Ejection Fraction
- Serum Creatinine
- Serum Sodium
- [Add other relevant features]

## Model Performance
- Accuracy: [Add accuracy]
- Precision: [Add precision]
- Recall: [Add recall]
- F1 Score: [Add F1 score]
