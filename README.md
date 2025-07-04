# Software Salary Prediction

Predict software industry salaries using machine learning and a user-friendly Flask web app.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Data Cleaning & Preparation](#data-cleaning--preparation)
- [Model Building & Evaluation](#model-building--evaluation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Web App (Flask Deployment)](#web-app-flask-deployment)
- [How to Run Locally](#how-to-run-locally)
- [Results & Insights](#results--insights)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview
This project predicts the salaries of software professionals based on various features such as company, job title, location, employment status, and more. It combines thorough data analysis, robust machine learning, and an interactive web interface for easy predictions.

## Features
- **End-to-end ML pipeline:** Data cleaning, EDA, feature engineering, model training, and evaluation.
- **Multiple regression models:** Linear Regression, Decision Tree, Random Forest, XGBoost.
- **Hyperparameter tuning:** RandomizedSearchCV for XGBoost.
- **Interactive Flask web app:** User-friendly interface for salary prediction.

## Dataset
- **Source:** [Kaggle - Software Professional Salaries 2022](https://www.kaggle.com/datasets/iamsouravbanerjee/software-professional-salaries-2022)
- **Shape:** 22,770 rows × 8 columns
- **Features:**
  - Rating
  - Company Name
  - Job Title
  - Salary (target)
  - Salaries Reported
  - Location
  - Employment Status
  - Job Roles

## Exploratory Data Analysis (EDA)
- **Univariate & Bivariate Analysis:**
  - Distribution plots for salary and rating
  - Job role and location frequency
  - Boxplots for salary by employment status and job role
  - Pairplots for numerical features
- **Key Insights:**
  - Salary distributions are right-skewed
  - Some job roles and companies dominate the dataset
  - Outliers and missing values identified and handled

## Data Cleaning & Preparation
- Dropped rows with missing company names
- Removed extreme outlier in salary
- Grouped rare categories in company and job title as 'Other'
- Standardized numerical features and one-hot encoded categorical features

## Model Building & Evaluation
- **Models Trained:**
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - XGBoost Regressor
- **Evaluation Metrics:** MAE, MSE, RMSE, R²
- **Best Model:** XGBoost (after hyperparameter tuning)

| Model              | MAE      | RMSE     | R²      |
|--------------------|----------|----------|---------|
| Linear Regression  | 356,296  | 539,130  | 0.24    |
| Decision Tree      | 426,012  | 702,916  | -0.30   |
| Random Forest      | 378,252  | 590,131  | 0.09    |
| XGBoost            | 349,637  | 532,842  | 0.25    |

## Hyperparameter Tuning
- Used `RandomizedSearchCV` for XGBoost with a wide parameter grid
- Best parameters improved R² to ~0.25 on the test set

## Web App (Flask Deployment)
- **Frontend:** Simple forms for user input (company, job title, location, etc.)
- **Backend:**
  - Loads the trained model (`Software Industry Salary Prediction.pkl`)
  - Accepts user input, preprocesses it, and predicts salary
  - Displays the predicted salary on a results page
- **Templates:**
  - `index.html`: Home page
  - `predict.html`: Input form
  - `result.html`: Prediction output

## How to Run Locally
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/software-salary-prediction.git
   cd software-salary-prediction
   ```
2. **Create a virtual environment and activate it:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Flask app:**
   ```bash
   python app.py
   ```
5. **Open your browser and go to:**
   [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Results & Insights
- XGBoost performed best, but all models had limited R², suggesting salary is influenced by additional factors not in the dataset.
- The app provides quick, accessible salary predictions for various roles and companies.

## Contributing
Contributions are welcome! Please open issues or pull requests for improvements, bug fixes, or new features.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements
- [Kaggle Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/software-professional-salaries-2022)
- [Scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/), [Flask](https://flask.palletsprojects.com/)
- Project by Yuva Yashvin, Yuvan Bharathi, Ritvik Marwah

---

*For questions or feedback, please contact the project maintainers.*