# Estimated Delivery Date (EDD) Prediction

## ClickPost Data Science Internship Assignment

---

## Table of Contents
1. Project Overview
2. Problem Statement
3. Deliverables
4. Evaluation Metrics
5. Dataset Description
6. Installation Instructions
7. Usage
8. Model Description
9. Variable Importance & Trends
10. Data Visualization
11. Engineering Documentation
12. Conclusion
13. Contact

---

## Project Overview

Logistics is a rapidly growing sector, with millions of orders dispatched daily. Accurately predicting the Estimated Delivery Date (EDD) is both challenging and crucial for enhancing customer satisfaction and optimizing logistics operations. This project focuses on developing a machine learning model to predict the EDD for orders of an e-commerce enterprise, based on historical shipment data.

---

## Problem Statement

**Objective:**  
Predict the Estimated Delivery Date (EDD) for each order of an e-commerce company.

**Definition of EDD:**  
EDD is defined as the number of days between the shipment date and the order delivery date.

**Context:**  
The training dataset comprises daily shipment data of an e-commerce enterprise from June 2022 to August 2022. The goal is to predict the `predicted_exact_sla` for shipments in the test dataset covering a subsequent 3-week period.

---

## Deliverables

1. **submission.csv:**  
   Contains the predicted EDD values corresponding to the `id` column in the test dataset.

2. **Model Description:**  
   A concise explanation of the machine learning model used, including its performance metrics and variable importance.

3. **Data Visualization:**  
   Visual representations highlighting important and useful patterns in the data.

4. **Code:**  
   The complete codebase (Jupyter Notebook) utilized for data preprocessing, model training, evaluation, and prediction generation.

5. **Engineering Documentation:**  
   A brief document outlining suggestions on integrating the EDD prediction model with the ClickPost system, enabling EDD consumption at both order and bulk levels.

---

## Evaluation Metrics

The submission will be evaluated based on the following parameters:

- **Accuracy of Predicted Values:**  
  The closeness of the predicted EDD values to the actual observed values.

- **Root Mean Squared Error (RMSE):**  
  Measures the average magnitude of the prediction errors, emphasizing larger errors.

- **Additional Metrics:**  
  - **Mean Absolute Error (MAE):** Average of absolute differences between predictions and actual values.
  - **R² Score:** Proportion of variance in the dependent variable predictable from the independent variables.
  - **Classification Metrics (if applicable):**  
    - **Accuracy**
    - **Precision**
    - **Recall**

---

## Dataset Description

The dataset consists of three files:

1. **train_.csv:**  
   - **Description:** Contains historical shipment data used to train the predictive model.
   - **Time Frame:** June 2022 to August 2022.
   - **Key Columns:**  
     - `id`: Unique identifier for each shipment.
     - `order_shipped_date`: Date when the order was shipped.
     - `order_delivered_date`: Date when the order was delivered.
     - `courier_partner_id`: Identifier for the courier partner.
     - `account_type_id`: Type identifier for the account.
     - `drop_pin_code`: PIN code where the order is to be delivered.
     - `pickup_pin_code`: PIN code from where the order is picked up.
     - `quantity`: Number of items in the order.
     - `account_mode`: Mode of the account (e.g., Air, Ground).
     - `order_delivery_sla`: Service Level Agreement (SLA) in days.

2. **test_.csv:**  
   - **Description:** Contains shipment data for which EDD predictions are to be made.
   - **Key Columns:** Similar to `train_.csv` but without the `order_delivered_date` and `order_delivery_sla`.

3. **pincodes.csv:**  
   - **Description:** Provides geographic details corresponding to each PIN code.
   - **Key Columns:**  
     - `Pincode`: Numeric postal code.
     - `CircleName`, `RegionName`, `DivisionName`, `OfficeName`: Geographic classifications.
     - `Latitude`, `Longitude`: Geographical coordinates.

---

## Installation Instructions

To replicate the project environment, follow these steps:

1. **Clone the Repository:**  
   ```bash
   git clone https://github.com/meakc/assignment.git
   cd assignment
   ```

2. **Set Up a Virtual Environment (Optional but Recommended):**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Required Packages:**  
   ```bash
   pip install -r requirements.txt
   ```
   *If a `requirements.txt` file is not provided, you can install the necessary packages manually:*  
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

---

## Usage

1. **Ensure Data Files are Available:**  
   Place `train_.csv`, `test_.csv`, and `pincodes.csv` in the project directory.

2. **Open the Jupyter Notebook:**  
   Launch Jupyter Notebook or JupyterLab and open `EDD_Prediction.ipynb`.

3. **Run the Notebook Cells:**  
   Execute each cell sequentially to perform data preprocessing, model training, evaluation, and prediction generation.

4. **Generate `submission.csv`:**  
   After running all cells, a `submission.csv` file will be created containing the predicted EDD values.

5. **Download the Trained Model (Optional):**  
   A download link will be provided within the notebook to download the serialized model (`random_forest_regressor.pkl`).

---

## Model Description

**Algorithm Used:**  
Random Forest Regressor

**Rationale:**  
Random Forest is an ensemble learning method known for its robustness and ability to handle both numerical and categorical features. It effectively manages feature interactions and reduces the risk of overfitting, making it suitable for predicting continuous variables like EDD.

**Model Performance:**  
- **RMSE:** 0.76  
- **MAE:** 0.36  
- **R² Score:** 0.81

These metrics indicate a strong correlation between the predicted and actual EDD values, showcasing the model's effectiveness.

---

## Variable Importance & Trends

The Random Forest model provides insights into the importance of each feature in predicting EDD. The most significant variables influencing the predictions are:

1. **courier_partner_id:**  
   Importance: 0.3735  
   *Indicates the courier partner's efficiency and reliability.*

2. **drop_pin_code:**  
   Importance: 0.3663  
   *Reflects geographic factors influencing delivery times.*

3. **order_shipped_date:**  
   Importance: 0.1469  
   *Captures temporal patterns and seasonality in deliveries.*

4. **pickup_pin_code:**  
   Importance: 0.0878  
   *Represents the origin location's impact on delivery speed.*

5. **account_mode:**  
   Importance: 0.0185  
   *Differentiates between various account types, such as Air or Ground.*

6. **account_type_id:**  
   Importance: 0.0070  
   *Classifies the account type, though it has minimal impact.*

7. **quantity:**  
   Importance: 0.0000  
   *Surprisingly, the number of items in the order does not influence EDD.*

---

## Data Visualization

Several visualizations have been created to understand the data and model performance:

1. **Feature Importance Bar Plot:**  
   Highlights the significance of each feature in the model.

3. **SLA Distribution Histogram:**  
   Shows the distribution of predicted SLA values.

4. **Residual Distribution Plot:**  
   Illustrates the residuals to assess the model's prediction errors.

5. **Confusion Matrix:**  
   (Applicable if classification metrics are used) Displays the performance of the classification model.

6. **Classification Report Heatmap:**  
   Visualizes precision, recall, and F1-score for each class.

7. **Class Distribution Plots:**  
   Compares the distribution of true and predicted classes.

8. **Performance Comparison Bar Plot:**  
   Compares regression and classification metrics side by side.

*All visualizations are generated using Seaborn and Matplotlib and are embedded within the Jupyter Notebook.*

---

## Engineering Documentation

### Integration with ClickPost System

To seamlessly integrate the EDD prediction model with the ClickPost system, consider the following steps:

1. **API Deployment:**  
   - **Containerization:** Use Docker to containerize the trained model and its dependencies.
   - **API Framework:** Deploy the model using frameworks like Flask or FastAPI to create RESTful endpoints.
   - **Endpoints:**  
     - **Single Order Prediction:** An endpoint to receive order details and return the predicted EDD.
     - **Bulk Prediction:** An endpoint to handle batch requests for bulk EDD predictions.

2. **Model Serving:**  
   - **Scalability:** Utilize platforms like AWS SageMaker, Google AI Platform, or Azure ML for scalable model serving.
   - **Monitoring:** Implement monitoring to track model performance and latency.

3. **Data Pipeline Integration:**  
   - **Real-Time Data:** Ensure that real-time shipment data is fed into the model for immediate EDD predictions.
   - **Batch Processing:** For bulk predictions, integrate with existing batch processing workflows.

4. **Security and Compliance:**  
   - **Authentication:** Secure the API endpoints with authentication mechanisms.
   - **Data Privacy:** Ensure compliance with data privacy regulations when handling shipment data.

5. **User Interface (UI):**  
   - **Dashboard:** Develop dashboards to visualize EDD predictions, model performance, and other key metrics for internal stakeholders.
   - **Notifications:** Integrate EDD predictions with notification systems to inform customers about their order status.

6. **Continuous Integration and Deployment (CI/CD):**  
   - **Automation:** Set up CI/CD pipelines to automate model retraining, testing, and deployment as new data becomes available.

7. **Documentation and Training:**  
   - **User Guides:** Provide comprehensive documentation for end-users on how to interpret and utilize EDD predictions.
   - **Training Sessions:** Conduct training for the logistics team to understand the model's capabilities and limitations.

### Suggestions for Enhancement

- **Hyperparameter Tuning:**  
  Optimize model parameters using techniques like Grid Search or Random Search to improve performance.

- **Feature Engineering:**  
  Incorporate additional features such as weather data, holidays, or traffic conditions that may affect delivery times.

- **Alternative Models:**  
  Experiment with other regression algorithms like Gradient Boosting, XGBoost, or Neural Networks to potentially enhance prediction accuracy.

- **Ensemble Methods:**  
  Combine predictions from multiple models to leverage their individual strengths.

- **Model Explainability:**  
  Utilize tools like SHAP or LIME to provide deeper insights into model predictions, aiding in transparency and trust.

---

## Conclusion

This project successfully developed a Random Forest Regressor model to predict the Estimated Delivery Date (EDD) for an e-commerce company's shipments. The model demonstrates strong performance with an R² score of approximately 0.81, indicating a significant ability to predict EDD based on the provided features. Feature importance analysis revealed that the courier partner and drop PIN code are the most influential factors affecting delivery times. Future enhancements can focus on integrating additional data sources, optimizing the model through hyperparameter tuning, and deploying the model within the ClickPost system for real-time EDD predictions.

---

## Contact

**Abhishek Kumar Choudhary**
