
# FinPay Customer Churn Prediction

## Description

This project aims to predict customer churn for the FinPay fintech app. Currently, sending notifications and discount offers to churn-prone customers is done manually. The goal of this project is to automate this process by building a machine learning model that classifies customers as **churn** or **not churn**, allowing marketing to automatically send targeted 10% discount push notifications to at-risk customers.

The model uses the following features:

- `wallet_id`, `onboard_date`, `user_class`, `state`, `region`
- `last_trans_date`, `first_trans_date`, `days_since_last_txn`, `days_to_first_txn`, `tenure_days`
- `total_txn_count`, `total_tpv`, `avg_tpv`, `txn_count_30d`, `tpv_30d`, `avg_txn_value_30d`
- `txn_count_90d`, `tpv_90d`, `avg_txn_value_90d`, `txn_trend`, `tpv_trend`, `avg_days_between_txn`
- `pos_share`, `transfer_share`, `digital_service_share`, `loan_share`, `deposit_share`
- `total_commission`, `success_rate`, `churn_flag`

The model uses **Logistic Regression** to classify customers, and the project is designed to be used in an automated pipeline for real-time churn prediction.

---

## Libraries Used

```python
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
Project Structure
train.py — script to train the churn prediction model

predict.py — script to make predictions on new customer data

serve.py — script to serve the model as an API

Pipfile & Pipfile.lock — pipenv environment files

Dockerfile — used to build the Docker image finpay_churn_serving:latest

Getting Started
1. Clone the repository
bash
Copy code
git clone <your-repo-url>
cd <your-repo-folder>
2. Activate the Python environment
bash
Copy code
pipenv install
pipenv shell
3. Train the model
bash
Copy code
python train.py
This will train the logistic regression model and save it as a pickle file for later use.

4. Make predictions
bash
Copy code
python predict.py --input data/new_customers.csv --output data/predictions.csv
5. Serve the model as an API
bash
Copy code
python serve.py
The API can then receive customer data and return churn predictions in real-time.

Docker Usage
To build the Docker image:

bash
Copy code
docker build -t finpay_churn_serving:latest .
To run the Docker container:

bash
Copy code
docker run -p 8000:8000 finpay_churn_serving:latest
The model will be served at http://localhost:8000.

Goal
By deploying this service, the FinPay data and engineering teams can automate customer churn detection and marketing campaigns, ensuring timely push notifications with 10% discount offers to reduce churn.

yaml
Copy code

---

If you want, I can also make a **slimmer, more marketing-friendly version** that you could attach to your GitHub repo or project documentation. It’ll be easier for stakeholders to read.  

Do you want me to do that?
