import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# ======================================
# AQI Category Function
# ======================================

def get_aqi_category(aqi):

    if aqi <= 50:
        return "Good 🟢"

    elif aqi <= 100:
        return "Satisfactory 🟡"

    elif aqi <= 200:
        return "Moderate 🟠"

    elif aqi <= 300:
        return "Poor 🔴"

    elif aqi <= 400:
        return "Very Poor 🟣"

    else:
        return "Severe ⚫"


# ======================================
# Load Files
# ======================================

model = joblib.load('../model/aqi_model.pkl')

encoder = joblib.load('../model/city_encoder.pkl')

importance = pd.read_csv(
    '../model/feature_importance.csv'
)

corr = pd.read_csv(
    '../model/correlation.csv',
    index_col=0
)

df = pd.read_csv(
    '../dataset/city_day.csv'
)

df['Date'] = pd.to_datetime(df['Date'])


# ======================================
# Title Section
# ======================================

st.title("🌍 Air Quality Index Prediction System")

st.info(
    "Machine Learning Model: XGBoost Regressor"
)

st.write(
    "Predict AQI using pollutant concentration levels."
)


# ======================================
# Metrics
# ======================================

col1, col2, col3 = st.columns(3)

col1.metric(
    "Cities",
    df['City'].nunique()
)

col2.metric(
    "Records",
    len(df)
)

col3.metric(
    "Average AQI",
    round(df['AQI'].mean(), 2)
)


# ======================================
# Sidebar Inputs
# ======================================

st.sidebar.header("Input Parameters")

city = st.sidebar.selectbox(
    "Select City",
    encoder.classes_
)

pm25 = st.sidebar.number_input(
    "PM2.5",
    min_value=0.0
)

pm10 = st.sidebar.number_input(
    "PM10",
    min_value=0.0
)

no = st.sidebar.number_input(
    "NO",
    min_value=0.0
)

no2 = st.sidebar.number_input(
    "NO2",
    min_value=0.0
)

nox = st.sidebar.number_input(
    "NOx",
    min_value=0.0
)

nh3 = st.sidebar.number_input(
    "NH3",
    min_value=0.0
)

co = st.sidebar.number_input(
    "CO",
    min_value=0.0
)

so2 = st.sidebar.number_input(
    "SO2",
    min_value=0.0
)

o3 = st.sidebar.number_input(
    "O3",
    min_value=0.0
)

benzene = st.sidebar.number_input(
    "Benzene",
    min_value=0.0
)

toluene = st.sidebar.number_input(
    "Toluene",
    min_value=0.0
)

xylene = st.sidebar.number_input(
    "Xylene",
    min_value=0.0
)

year = st.sidebar.number_input(
    "Year",
    value=2026
)

month = st.sidebar.number_input(
    "Month",
    min_value=1,
    max_value=12,
    value=6
)

day = st.sidebar.number_input(
    "Day",
    min_value=1,
    max_value=31,
    value=1
)


# ======================================
# Prediction Section
# ======================================

if st.button("Predict AQI"):

    city_code = encoder.transform([city])[0]

    sample = pd.DataFrame({
        'City': [city_code],
        'PM2.5': [pm25],
        'PM10': [pm10],
        'NO': [no],
        'NO2': [no2],
        'NOx': [nox],
        'NH3': [nh3],
        'CO': [co],
        'SO2': [so2],
        'O3': [o3],
        'Benzene': [benzene],
        'Toluene': [toluene],
        'Xylene': [xylene],
        'Year': [year],
        'Month': [month],
        'Day': [day]
    })

    prediction = model.predict(sample)

    aqi = prediction[0]

    st.success(
        f"Predicted AQI: {aqi:.2f}"
    )

    category = get_aqi_category(aqi)

    st.subheader(
        f"Category: {category}"
    )

    if aqi <= 50:
        st.success("Air Quality is Good")

    elif aqi <= 100:
        st.info("Air Quality is Satisfactory")

    elif aqi <= 200:
        st.warning("Air Quality is Moderate")

    else:
        st.error("Air Quality is Poor")


# ======================================
# Feature Importance
# ======================================

st.subheader("📊 Feature Importance")

st.bar_chart(
    importance.set_index('Feature')
)


# ======================================
# AQI Trend
# ======================================

st.subheader("📈 AQI Trend Over Time")

trend = df.groupby('Date')['AQI'].mean()

st.line_chart(trend)


# ======================================
# City Dashboard
# ======================================

st.subheader("🏙️ City-wise AQI Dashboard")

selected_city = st.selectbox(
    "Select City",
    sorted(df['City'].unique())
)

city_df = df[
    df['City'] == selected_city
]

city_df = city_df.sort_values(
    'Date'
)

st.line_chart(
    city_df.set_index('Date')['AQI']
)


# ======================================
# Correlation Heatmap
# ======================================

st.subheader("🔥 Correlation Heatmap")

fig, ax = plt.subplots(
    figsize=(10, 6)
)

sns.heatmap(
    corr,
    cmap='coolwarm',
    annot=False,
    linewidths=0.5,
    ax=ax
)

st.pyplot(fig)


# ======================================
# Dataset Preview
# ======================================

st.subheader("📄 Dataset Preview")

st.dataframe(
    df.head()
)