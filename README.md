---

# **Car Price Prediction Using Machine Learning**

## **Overview**

In the dynamic used car market, accurately predicting vehicle prices is crucial for enhancing customer satisfaction and optimizing sales strategies. This repository presents a comprehensive machine learning project focused on predicting the prices of used cars based on various features such as make, model, year, fuel type, transmission type, mileage, and more. Leveraging historical data from CarDekho, this project aims to develop a robust, user-friendly tool that aids sales representatives and customers alike. The final output is a deployed Streamlit application, enabling users to input car details and receive instant price predictions.

## **Project Structure**

The repository is organized as follows:

1. **Jupyter Notebook**  
   - **File**: `CarDheko.ipynb`  
   - **Description**: This notebook contains the entire workflow for data cleaning, feature engineering, model selection, training, and evaluation. It includes an exploratory data analysis (EDA) to identify key features influencing car prices and implements multiple machine learning models such as Linear Regression, Decision Trees, Random Forest, and Gradient Boosting with hyperparameter tuning to achieve optimal performance.

2. **Streamlit Application**  
   - **File**: `car_dekho_app.py`  
   - **Description**: A Streamlit-based web application providing an interactive interface for users to input car specifications and obtain a predicted price. The app is designed for ease of use, making it accessible to both technical and non-technical users.

3. **Project Report**  
   - **File**: `project_report.pdf`  
   - **Description**: A detailed report that documents the entire project lifecycle, from the initial problem statement and data preprocessing to model evaluation and deployment. It includes rationale for the chosen methodologies, a summary of results, and insights derived from the analysis.

4. **User Guide**  
   - **File**: `user_guide.pdf`  
   - **Description**: A comprehensive guide providing step-by-step instructions on how to use the Streamlit application. It explains how to navigate the app, input data, and interpret the results, ensuring it is user-friendly for individuals of all technical backgrounds.

5. **Resources**  
   - **File**: `resources.zip`  
   - **Contents**: This ZIP file includes all necessary resources, such as the cleaned dataset used for training, serialized model files (joblib), and any additional materials that support the project.

## **Getting Started**

### **Prerequisites**

To run the Jupyter Notebook and Streamlit application, the following software and libraries are required:

- **Python 3.7+**
- **Required Libraries**:  
  ```
  pip install pandas numpy scikit-learn matplotlib seaborn streamlit
  ```

### **Running the Streamlit Application**

1. **Input Fields**: The application allows users to input various car attributes, including make, model, year, fuel type, transmission type, and mileage.
2. **Price Prediction**: After entering the details, click the **'Predict'** button to receive the estimated price of the car.
3. **User Interface**: The interface is designed to be intuitive, ensuring a smooth experience for both tech-savvy and non-technical users.

### **Model Training and Evaluation**

1. **Data Cleaning**: The dataset undergoes thorough cleaning to address missing values, encode categorical variables, and scale numerical features appropriately.
2. **Model Selection**: Multiple models, including Linear Regression, Decision Trees, Random Forest, and Gradient Boosting, are trained and evaluated. Hyperparameter tuning is conducted to optimize model performance.
3. **Evaluation Metrics**: Models are evaluated using Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²) values to determine the best model for deployment.

## **Results**

- **Best Model**: The model with the highest R² score and the lowest MSE and MAE was selected for deployment.
- **Predictive Accuracy**: The final model demonstrated strong predictive accuracy, making it reliable for practical applications in estimating used car prices.

## **Repository Contents**

- `car_dekho_cleaning_and_modeling.ipynb`: Jupyter Notebook for data processing, model training, and evaluation.
- `car_dekho_app.py`: Streamlit application script for real-time price prediction.
- `project_report.pdf`: Comprehensive project report covering all aspects of the project.
- `user_guide.pdf`: Step-by-step guide for using the Streamlit application.
- `resources.zip`: Contains the cleaned dataset, model files, and additional resources.

## **Acknowledgements**

This project utilizes data from CarDekho and leverages several open-source libraries, including Scikit-learn for machine learning and Streamlit for application deployment.


## **References**

- [CarDekho](https://www.cardekho.com)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Streamlit Documentation](https://docs.streamlit.io)

## **Contact**

For any questions or suggestions, please feel free to reach out at [mryunus.in@gmail.com].

---
