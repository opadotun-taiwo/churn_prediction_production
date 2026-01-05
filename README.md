
# FinPay Customer Churn Prediction
![Churn Prediction Architecture](https://cdn.sanity.io/images/pghoxh0e/production/9dd57f643d87242fb736f32fe109cb2c327b1205-960x540.png?rect=0,18,960,504&w=1200&h=630)

## Description

This project aims to predict customer churn for the FinPay fintech app. Currently, sending notifications and discount offers to churn-prone customers is done manually by downloading records from dashboard.

# The goal of this project is to automate customer churn detection by building a machine learning model that classifies customers as “churn” or “not churn.” The model is deployed as an API on AWS Elastic Beanstalk, enabling internal developers and systems to automatically identify at-risk customers and trigger targeted 10% bill payment discount push notifications to reduce churn.

## What this project does

This project demonstrates how machine learning can be used to predict customer churn in real time and integrate those predictions directly into business workflows.

A Logistic Regression model is trained to analyze customer behavior and determine whether a customer is likely to churn. Once trained, the model is exposed as a REST API, allowing other services (such as marketing or notification systems) to request churn predictions instantly.

# This makes it possible to automatically:

Detect customers at risk of leaving

Send timely, personalized incentives (e.g. a 10% discount)

Reduce churn without manual intervention

# Model Details

Algorithm: Logistic Regression

Use case: Binary classification (Churn vs Not Churn)

Deployment: AWS Elastic Beanstalk

Prediction mode: Real-time via API

The model is designed to fit seamlessly into an automated production pipeline, rather than being a one-off analysis.

# Libraries Used
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Project Structure

train.py — Trains the churn prediction model

predict.py — Generates churn predictions for new customer data

serve.py — Serves the trained model as an API

Pipfile / Pipfile.lock — Python dependency management

Dockerfile — Builds the Docker image (finpay_churn_serving:latest)

# Getting Started
1️ Clone the repository
git clone <your-repo-url>
cd <your-repo-folder>

2️ Set up the environment
pipenv install
pipenv shell

3️ Train the model
python train.py


This trains the Logistic Regression model and saves it as a pickle file for reuse.

4️ Make predictions on new data
python predict.py --input data/new_customers.csv --output data/predictions.csv

5️ Serve the model as an API
python serve.py


The API will accept customer data and return churn predictions in real time.

 Docker Usage
Build the image
docker build -t finpay_churn_serving:latest .

Run the container
docker run -p 8000:8000 finpay_churn_serving:latest


The service will be available at:

http://localhost:8000

# Business Impact

By deploying this service, FinPay’s data and engineering teams can:

Automate churn detection

Reduce customer drop-off through timely incentives

Power marketing campaigns with real-time ML predictions

Eliminate manual churn analysis
