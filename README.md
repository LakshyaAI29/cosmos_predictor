# cosmos_predictor
Machine Learning model to predict the budget and success rate of space missions based on mission type, satellite type, technology used, and mission duration. Built using Python, pandas, and scikit-learn.

# Space Mission Budget & Success Predictor

## Description
This project uses a *Linear Regression model* to predict the *budget (in Billion $)* and *success rate (%)* of space missions. The model considers:  
- Mission Type  
- Satellite Type  
- Technology Used  
- Duration of the mission  

It demonstrates *data preprocessing, **one-hot encoding, **train-test split, model evaluation metrics (MAE, MSE, R²), and **real-time predictions based on user input*.

---

## Features
- Predicts *budget* and *success rate* of space missions.  
- Handles *categorical features* using one-hot encoding.  
- Evaluates model performance using:  
  - *Mean Absolute Error (MAE)*  
  - *Mean Squared Error (MSE)*  
  - *R² Score*  
- Allows users to *input their own mission parameters* and get predictions instantly.  

---

## Dataset
- Contains historical space mission data including budget, mission type, satellite type, technology used, and duration.  
- Originally sourced from: Global_Space_Exploration_Dataset.csv

---

## Installation & Requirements
1. Clone the repository:  
```bash
git clone <your-repo-url>
