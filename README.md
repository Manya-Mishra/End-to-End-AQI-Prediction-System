# 🌍 Air Quality Index (AQI) Prediction System

## 📌 Project Overview

The Air Quality Index (AQI) Prediction System is a Machine Learning-based web application developed to predict air quality levels using environmental and pollutant data. The project analyzes key atmospheric pollutants and predicts the AQI value, helping users understand air quality conditions and potential environmental risks.

The application is built using Python, XGBoost Regression, and Streamlit, providing an interactive dashboard for AQI prediction and visualization.

---

## 🎯 Objectives

* Predict Air Quality Index (AQI) using pollutant concentration data.
* Analyze the impact of various pollutants on air quality.
* Visualize AQI trends and pollutant relationships.
* Provide an interactive web interface for real-time AQI prediction.
* Demonstrate an end-to-end Machine Learning workflow from data preprocessing to deployment.

---

## 📂 Dataset

**Dataset Source:** Kaggle – Air Quality Data in India

The dataset contains city-wise air quality measurements collected across various Indian cities and includes pollutant concentrations such as:

* PM2.5
* PM10
* NO
* NO₂
* NOx
* NH₃
* CO
* SO₂
* O₃
* Benzene
* Toluene
* Xylene

Target Variable:

* AQI (Air Quality Index)

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Joblib
* Streamlit

### Development Tools

* Jupyter Notebook
* Anaconda
* VS Code

---

## ⚙️ Machine Learning Pipeline

### 1. Data Collection

* Imported AQI dataset from Kaggle.
* Loaded city-wise air quality measurements.

### 2. Data Preprocessing

* Handled missing values.
* Removed irrelevant columns.
* Converted date features into Year, Month, and Day.
* Encoded categorical city data using Label Encoding.

### 3. Exploratory Data Analysis (EDA)

* AQI distribution analysis.
* Pollutant correlation analysis.
* City-wise AQI comparison.
* Trend analysis over time.

### 4. Feature Engineering

* Extracted temporal features from date columns.
* Selected important pollutant attributes.

### 5. Model Building

Implemented and compared multiple machine learning algorithms:

* Linear Regression
* Random Forest Regressor
* XGBoost Regressor

### 6. Model Evaluation

Performance evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

### 7. Model Saving

Saved trained artifacts using Joblib:

* aqi_model.pkl
* city_encoder.pkl

---

## 📊 Dashboard Features

### AQI Prediction

Predict AQI based on pollutant concentrations and city selection.

### AQI Category Classification

Displays AQI category:

* Good
* Satisfactory
* Moderate
* Poor
* Very Poor
* Severe

### Feature Importance Analysis

Visualizes the contribution of pollutants to AQI prediction.

### AQI Trend Visualization

Shows AQI variations over time.

### City-wise AQI Dashboard

Provides city-specific AQI trend analysis.

### Correlation Heatmap

Displays relationships among pollutants and AQI.

### Dataset Preview

Interactive dataset exploration within the dashboard.

---

## 📁 Project Structure

```text
AQI_Prediction_Project
│
├── dataset
│   └── city_day.csv
│
├── model
│   ├── aqi_model.pkl
│   ├── city_encoder.pkl
│   ├── feature_importance.csv
│   └── correlation.csv
│
├── notebooks
│   └── AQI_Prediction.ipynb
│
├── app
│   └── app.py
│
└── README.md
```

---

## 🚀 How to Run the Project

### Clone Repository

```bash
git clone <repository-url>
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit Application

```bash
cd app
streamlit run app.py
```

---

## 📈 Future Enhancements

* Real-time AQI prediction using live APIs.
* Weather data integration.
* AQI forecasting for future dates.
* Geographical visualization using maps.
* Cloud deployment and monitoring.

---

## 💡 Key Learnings

* Data Cleaning and Preprocessing
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Regression Modeling
* Model Evaluation Techniques
* Interactive Dashboard Development
* Machine Learning Model Deployment

---

## 👨‍💻 Author

**Manya Mishra**

Aspiring Data Analyst | Machine Learning Enthusiast | Python Developer

This project demonstrates an end-to-end machine learning workflow, from raw environmental data analysis to deployment of a production-ready prediction dashboard.
