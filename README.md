# Mall Customer Segmentation API using FastAPI and Docker

## Project Overview

This project demonstrates the deployment of a Machine Learning model using FastAPI and Docker.

The model predicts whether a customer belongs to a **High Value Customer Segment** or a **Low Value Customer Segment** based on the following features:

* Gender
* Age
* Annual Income
* Spending Score

The trained Random Forest model is exposed through a REST API built with FastAPI and containerized using Docker for easy deployment.

---

## Dataset

**Dataset:** Mall Customers Dataset

Features:

* Genre
* Age
* Annual_Income_(k$)
* Spending_Score

Target Variable:

A new target column named **Segment** is generated using:

```python
df["Segment"] = (df["Spending_Score"] >= 50).astype(int)
```

Where:

* 1 = High Value Customer
* 0 = Low Value Customer

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* FastAPI
* Uvicorn
* Docker
* Joblib

---

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Train the Model

Run:

```bash
python train_model.py
```

Output:

```text
Accuracy: 1.0
Model saved successfully!
```

This generates:

```text
model.pkl
```

---

## Run the API Locally

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

Open Swagger Documentation:

```text
http://127.0.0.1:8000/docs
```

---

## API Endpoint

### POST /predict

### Request

```json
{
  "gender": 1,
  "age": 25,
  "annual_income": 60,
  "spending_score": 80
}
```

### Response

```json
{
  "segment": 1,
  "segment_name": "High Value Customer"
}
```

---

## Docker Deployment

### Build Docker Image

```bash
docker build -t mall-customer-api .
```

### Run Docker Container

```bash
docker run -p 8000:8000 mall-customer-api
```

Open:

```text
http://localhost:8000/docs
```

---

## Project Structure

```text
Mall_Customer_API/
│
├── Mall_Customers.csv
├── train_model.py
├── app.py
├── requirements.txt
├── Dockerfile
├── README.md
└── model.pkl
```

---

## Conclusion

This project successfully demonstrates the complete machine learning deployment workflow, including model training, API development using FastAPI, and containerization using Docker. The deployed API can be used to predict customer segments based on demographic and spending behavior.
