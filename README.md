# ecommerce-ml
This repository showcases a diverse set of machine learning and statistical modeling techniques for e-commerce.

* **Purpose** 
  - To highlight a diverse set of machine learning and statistical modeling techniques for e-commerce.
 
* **Predicting Customer Purchases Using Random Forests**
  - **Aim** - The aim of this project is to develop and evaluate a Random Forest Classifier to predict whether a customer makes a purchase on an e-commerce platform based on demographic and behavioral factors. Model outcomes are compared with LightGMB and XGBoost.
  - **Objectives** - (1) Identify key predictors of customer purchases, (2) assess the accuracy and performance of the model , and (3) Understand the impact of customer age, session duration, and promotion engagement on purchase decisions
  - **Hypothesis** - H₀: There is no significant relationship between customer features and purchasing behavior; H₁: Customer attributes (age, session duration, and promotion engagement) significantly influence purchases
  - **File:** [`random_forest_purchases.py`](https://github.com/h-hedman/ecommerce-ml/blob/main/logistic_regression_prediction_model_ecommerce.py)
* **Logistic Regression Prediction model of Revenue Conversion**
  - **Aim** - This model predicts revenue conversion in e-commerce using synthetic data. It leverages features such as impressions, clicks, and past conversions with a logistic regression framework. 
  - **Objectives** - Develop and evaluate a predictive model that accurately forecasts revenue conversion, 
  - **Hypothesis** - H₀: Impressions, clicks, and past conversion history do not significantly influence revenue conversion.; H₁: Higher impressions, increased clicks, and positive past conversion history significantly boost the likelihood of revenue conversion
  - **File:** [`logistic_regression_prediction_model_ecommerce.py`](https://raw.githubusercontent.com/h-hedman/ecommerce-ml/refs/heads/main/logistic_regression_prediction_model_ecommerce.py)
* **LSTM-Based E-Commerce Revenue Forecasting**
  - **Aim** - This model predicts future e-commerce revenue based on historical sales data using a Long Short-Term Memory (LSTM) network. 
  - **Objectives** - (1) Develop and evaluate a predictive model that accurately forecasts e-commerce revenue based on past revenue patterns, (2) Identify key trends in sales behavior and capture seasonality effects in the data, (3) Utilize deep learning techniques (LSTM) to enhance forecasting accuracy over traditional statistical models 
  - **Hypothesis** - H₀: Past revenue patterns do not significantly influence future revenue; H₁: Historical revenue data, including trend and seasonality, significantly influences future revenue predictions
  - **File:** [`ecommerce_revenue_forecasting_lstm.py`](https://raw.githubusercontent.com/h-hedman/ecommerce-ml/refs/heads/main/ecommerce_revenue_forecasting_lstm.py)  
* **Profit Optimization Using Golden Ratio in E-Commerce Pricing**
  - **Aim** - This model applies the Golden Search alogirthm for optimal price selection. 
  - **Objectives** - (1) Identify the optimal price that maximizes profit in an e-commerce setting, (2) Understand how price sensitivity affects demand and revenue, and (3) Utilize the Golden Section Search algorithm to efficiently search for the best price  
  - **Hypothesis** - H₀: Price does not significantly impact demand and profit; H₁: There exists an optimal price that maximizes profit by balancing price sensitivity and unit costs
  - **File:** [`main.cpp`](https://raw.githubusercontent.com/h-hedman/ecommerce-ml/refs/heads/main/main.cpp)  
* **Compartmental Revenue Modeling (Visitors → Engaged → Purchasers)**
  - **Aim** - This model simulates the flow of users in an e-commerce setting using a compartmental system. 
  - **Objectives** - (1) Simulate and evaluate the transition of users from (Visitors → Engaged → Purchasers), (2) Quantify the effectiveness of engagement and purchase rates (`beta`, `gamma`), (3) Generate **actionable insights** into customer retention and revenue conversion dynamics.
  - **Hypothesis** - H₀: The engagement rate (V → E) and purchase rate (E → P) do not significantly affect the number of purchasers over time; H₁: Higher engagement and purchase rates significantly increase the number of users who convert to purchases
  - **File:** [`main.cpp`](https://raw.githubusercontent.com/h-hedman/ecommerce-ml/refs/heads/main/compartmental_model_consumer_trends.R)  



