# Import Packages
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
import joblib 
from sklearn.linear_model import LinearRegression 


# Import Seaborn
import streamlit as st

# Title of your Web App
st.title('Forecasting Sales')

# Describe your Web App
st.write("We demonstrate how we can forecast advertising sales based on ad expenditure.")

# Read Data
data = pd.read_csv('advertising_regression.csv')

# Show Data
data

# Let's Draw A Histogram

### TV
st.subheader('TV Ad Cost Distribution')

# Use numpy to generate bins for age
hist_values = np.histogram(data.TV, bins=300, range=(0,300))[0]

# Show Bar Chart
st.bar_chart(hist_values)

### Newspaper
st.subheader('Newspaper Ad Cost Distribution')

# Use numpy to generate bins for age
hist_values = np.histogram(data.newspaper, bins=300, range=(0,300))[0]

# Show Bar Chart
st.bar_chart(hist_values)

### Radio
st.subheader('Radio Ad Cost Distribution')

# Use numpy to generate bins for age
hist_values = np.histogram(data.radio, bins=300, range=(0,300))[0]

# Show Bar Chart
st.bar_chart(hist_values)

### Sales
st.subheader('Historical Sales Distribution')

# Use numpy to generate bins for age
hist_values = np.histogram(data.sales, bins=300, range=(0,300))[0]

# Show Bar Chart
st.bar_chart(hist_values)

# Add sliders and assign them to variables
st.sidebar.subheader('Advertising Costs')

# TV Slider
TV = st.sidebar.slider('TV Advertising Cost', 0, 300, 150) # (Title, min value, max value, default value)

# Radio Slider
radio = st.sidebar.slider('Radio Advertising Cost', 0, 50, 25) # (Title, min value, max value, default value)

# Newspaper Slider
newspaper = st.sidebar.slider('Newspaper Advertising Cost', 0, 250, 125) # (Title, min value, max value, default value)

# Load saved machine learning model
st.subheader("Predicted Sales")

# Load model using joblib
saved_model = joblib.load('adsales_linearregression.mdl')

# Predict sales using variables/features
predicted_sales = saved_model.predict([[TV, radio, newspaper]])[0]

# Print prediction
st.write(f"Predicted sales is {int(predicted_sales*1000)} dollars.")

