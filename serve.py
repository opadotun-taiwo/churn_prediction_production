import requests

#host= "churn-serving-env.eba-xm5mtzgr.us-east-1.elasticbeanstalk.com"
url = f'http://127.0.0.1:9696/predict'


customer = {
  "user_class": "App User",
  "region": "South West",
  "days_since_last_txn": 488.0,
  "days_to_first_txn": 276.0,
  "tenure_days": 764.0,
  "total_txn_count": 307,
  "total_tpv": 3108527.0,
  "avg_tpv": 10125.5,
  "txn_count_30d": 0,
  "tpv_30d": 0.0,
  "avg_txn_value_30d": 0.0,
  "txn_count_90d": 0,
  "tpv_90d": 0.0,
  "avg_txn_value_90d": 0.0,
  "txn_trend": 0.0,
  "tpv_trend": 0.0,
  "avg_days_between_txn": 0.9,
  "pos_share": 0.0,
  "transfer_share": 0.88,
  "digital_service_share": 0.013,
  "loan_share": 0.0,
  "deposit_share": 0.11,
  "total_commission": 2973295,
  "success_rate": 0.83
}


response = requests.post(url, json=customer)
print(response.json())