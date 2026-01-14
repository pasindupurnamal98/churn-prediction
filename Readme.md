# Customer Churn Prediction

## Project Overview

This project focuses on building a machine learning model to predict customer churn for a telecommunications company. Customer churn, the phenomenon of customers discontinuing their service, is a critical business problem. By accurately predicting which customers are likely to churn, businesses can proactively implement retention strategies, thereby reducing customer attrition and maximizing customer lifetime value. This repository contains the Jupyter Notebook (`Customer_Churn_Prediction.ipynb`) detailing the entire process from data exploration to model deployment.

## Features

The dataset used in this project includes various customer attributes, which can be broadly categorized as follows:

-   **Demographic Information**: Gender, Senior Citizen status, Partner status, Dependents status.
-   **Service Information**: Phone Service, Multiple Lines, Internet Service (DSL, Fiber Optic, No), Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies.
-   **Billing Information**: Contract type (Month-to-month, One year, Two year), Paperless Billing, Payment Method, Monthly Charges, Total Charges.
-   **Tenure**: The number of months the customer has stayed with the company.
-   **Churn**: The target variable, indicating whether the customer churned (Yes) or not (No).

## Installation

To set up the project locally, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd Customer-Churn-Prediction
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file would typically be generated from the notebook's imports. For this project, the key libraries include `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `imblearn`, and `xgboost`.)*

## Usage

1.  **Open the Jupyter Notebook**:
    ```bash
    jupyter notebook Customer_Churn_Prediction.ipynb
    ```

2.  **Run all cells**: Execute the cells sequentially to perform data loading, preprocessing, model training, and evaluation.

3.  **Explore the results**: Review the visualizations, model performance metrics, and the predictive system demonstration within the notebook.

## Model Performance

The project utilizes a **Random Forest Classifier** for churn prediction, which achieved the following performance metrics on the test dataset:

-   **Accuracy**: 0.7786
-   **Precision (Class 0 - No Churn)**: 0.85
-   **Recall (Class 0 - No Churn)**: 0.85
-   **F1-Score (Class 0 - No Churn)**: 0.85
-   **Precision (Class 1 - Churn)**: 0.58
-   **Recall (Class 1 - Churn)**: 0.59
-   **F1-Score (Class 1 - Churn)**: 0.58

These metrics indicate a reasonable performance, especially considering the class imbalance addressed using SMOTE. The model demonstrates good ability to identify non-churning customers, and a moderate ability to identify churning customers.

## Future Enhancements

Potential areas for further improvement include:

-   **Hyperparameter Tuning**: Optimize model parameters for better performance.
-   **Model Selection**: Experiment with other advanced machine learning models.
-   **Downsampling**: Explore downsampling techniques as an alternative to SMOTE.
-   **Addressing Overfitting**: Implement strategies to mitigate overfitting.
-   **Stratified K-Fold Cross-Validation**: Utilize more robust cross-validation techniques for model evaluation.

## License

This project is open-source and available under the MIT License. See the `LICENSE` file for more details.


