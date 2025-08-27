🚢 Titanic Survival Prediction & Exploratory Data Analysis (EDA)
📌 Project Overview

This project focuses on analyzing the famous Titanic dataset from Kaggle’s Titanic: Machine Learning from Disaster competition
.
The main goals are:

Perform Exploratory Data Analysis (EDA) to understand the dataset.

Clean and preprocess the data.

Build a machine learning model to predict passenger survival.

📂 Dataset

The dataset contains passenger information such as:

PassengerId – Unique ID

Pclass – Ticket class (1 = First, 2 = Second, 3 = Third)

Name, Sex, Age – Personal details

SibSp, Parch – Number of relatives aboard

Ticket, Fare, Cabin – Travel details

Embarked – Port of embarkation (C, Q, S)

Survived – Target variable (0 = No, 1 = Yes)

🔍 Exploratory Data Analysis (EDA)

During EDA, we aim to:

Check missing values (Age, Cabin, Embarked have missing values).

Understand distributions (e.g., Age distribution, Fare skewness).

Visualize survival trends:

Women and children had higher survival chances.

First-class passengers had better survival than third-class.

Passengers embarked from different ports show varying survival rates.

📊 Common plots:

Bar plots for categorical survival rates

Histograms for age and fare

Heatmaps for correlation

🛠️ Data Preprocessing

Handle missing values (impute Age, drop/encode Cabin, fill Embarked).

Encode categorical features (Sex, Embarked).

Feature engineering (e.g., FamilySize = SibSp + Parch + 1).

Scale numerical features (Age, Fare).

Split dataset into train/test sets.

🤖 Model Building

We experiment with multiple algorithms:

Logistic Regression

Decision Tree / Random Forest

Gradient Boosting (XGBoost, LightGBM)

Support Vector Machine

Evaluation metric: Accuracy / F1-score / ROC-AUC

📈 Results

Baseline Logistic Regression achieves ~78% accuracy.

Ensemble models (Random Forest, XGBoost) usually perform better (~80–83%).

🚀 How to Run

Clone this repo:

git clone https://github.com/your-username/titanic-survival-prediction.git
cd titanic-survival-prediction


Install requirements:

pip install -r requirements.txt


Run Jupyter Notebook:

jupyter notebook Titanic_EDA_Model.ipynb

📦 Requirements

Python 3.8+

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

XGBoost / LightGBM (optional)

Jupyter Notebook

📜 Future Work

Hyperparameter tuning for improved performance.

Use deep learning models for experimentation.

Feature engineering on ticket groups & cabin letters.

🙌 Acknowledgments

Kaggle Titanic Competition

Inspiration from Kaggle kernels & ML community
