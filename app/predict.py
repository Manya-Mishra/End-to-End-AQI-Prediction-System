import joblib
import pandas as pd

# Load saved model
model = joblib.load('../model/aqi_model.pkl')

# Load encoder
encoder = joblib.load('../model/city_encoder.pkl')

# Convert city name into numerical value
city_code = encoder.transform(['Delhi'])[0]

# New sample
sample = pd.DataFrame({
    'City':[city_code],
    'PM2.5':[85],
    'PM10':[150],
    'NO':[25],
    'NO2':[40],
    'NOx':[60],
    'NH3':[20],
    'CO':[1.5],
    'SO2':[10],
    'O3':[35],
    'Benzene':[5],
    'Toluene':[7],
    'Xylene':[1],
    'Year':[2026],
    'Month':[6],
    'Day':[3]
})

# Predict
prediction = model.predict(sample)

print("Predicted AQI:", prediction[0])