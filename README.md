
# Customer Churn Prediction

Hello Folks, This project predicts whether a customer will churn (leave) or not using telecom customer data. It uses a machine learning model trained on historical customer data and deployed with a **Streamlit** web app.

## Overview

Customer churn is a major concern for service providers, especially in the telecom industry. This ML-based solution classifies customers into churn and non-churn categories using important features like monthly charges, tenure, contract type, etc.


## Project Structure

Customer Churn Prediction Project.ipynb   # Jupyter Notebook with EDA, model training and evaluation
model.pkl                                 # Trained machine learning model (e.g., Logistic Regression, RandomForest)
scaler.pkl                                # Feature scaler for input normalization
streamlit_app.py                          # Streamlit-based frontend to make predictions
telecom_customer_churn.csv                # Telecom customer data (source dataset)
README.md                                 # Project documentation



## Tech Stack

- Python 3.x
- Pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn
- Streamlit
- Joblib / Pickle

## How to Run

### 1. Clone the repo

```bash
git clone https://github.com/satyamkurum/Customer_Churn_Prediction.git
cd Customer_Churn_Prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Streamlit app

```bash
streamlit run streamlit_app.py
```

You’ll see a local URL that opens the web interface for making predictions.

## Model Output

The trained model predicts:
- **Will Stay**
- **Will Churn**

Metrics like **accuracy, precision, recall, and F1-score** are computed in the notebook.

## License

This project is licensed under the [MIT License](LICENSE).

## Author

**Satyam Kurum**  
_Machine Learning & Web Enthusiast_  
[LinkedIn](https://linkedin.com/in/satyamkurum) | [GitHub](https://github.com/satyamkurum)

> Feel free to ⭐️ this repo and fork it for your own projects!
