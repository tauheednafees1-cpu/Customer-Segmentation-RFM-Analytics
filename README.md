# Customer Segmentation & RFM Analytics

An interactive Streamlit dashboard that analyzes customer purchasing behavior using RFM analysis, customer segmentation, risk scoring, CLV estimation, and retention campaign planning.

## Business Problem

Businesses need to understand customer purchasing behavior so they can identify:

- High-value customers
- Loyal customers
- Customers at risk of becoming inactive
- Lost customers
- Customers who need retention campaigns

This project converts transaction data into customer-level insights and actionable retention recommendations.

## Features

- RFM Analysis
- Recency, Frequency and Monetary distributions
- Customer segmentation
- Champions / Loyal / At Risk / Lost classification
- Customer risk scoring
- Customer 360 analysis
- CLV proxy calculation
- Revenue concentration analysis
- Retention campaign budget planning
- CSV upload support
- Interactive dashboard
- Data quality checks
- Downloadable customer segmentation data
- Downloadable retention plan

## Tech Stack

- Python
- Pandas
- NumPy
- Plotly
- Streamlit

## RFM Analysis

### Recency
Number of days since the customer's most recent purchase.

### Frequency
Number of unique orders made by the customer.

### Monetary
Total amount spent by the customer.

The project assigns each customer an RFM score from 1–5 for Recency, Frequency and Monetary value.

## Customer Segments

The application identifies customers such as:

- Champions
- Loyal
- New / Promising
- Big Spenders
- At Risk
- Lost
- Regular

Each segment is associated with a recommended business action.

## Risk Analysis

The project calculates an explainable customer risk score using:

- Purchase recency
- Expected purchase interval
- Order frequency
- Monetary value

The result is categorized into:

- Low Risk
- Medium Risk
- High Risk

## Customer 360

The Customer 360 section provides an individual customer view including:

- RFM score
- Customer segment
- Recency
- Order frequency
- Total spending
- Risk level
- Average Order Value
- CLV proxy
- Purchase history

## Retention Campaign Planner

The retention planner creates a hypothetical campaign budget allocation using:

- Customer lifetime value proxy
- Revenue contribution
- High-risk customer concentration

The dashboard provides a suggested budget percentage for each customer segment.

## Project Structure

```text
Customer-Segmentation-RFM-Analytics/
│
├── app.py
├── requirements.txt
├── README.md
├── generate_sample_data.py
├── .gitignore
│
├── data/
│   └── sample_transactions.csv
│
└── screenshots/
    ├── Executive Dashboard.png
    ├── RFM-analysis.png
    ├── Customer 360.png
    └── Retention Planner.png