import numpy as np 
import pandas as pd 
import os
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', font_scale=5)

@st.cache
def load_data():
    return pd.read_csv('insurance_regression.csv')

data = load_data()

### Set Title
st.title("Insurance Pricing App")
st.write("""From the data below, we built a machine learning-based pricing model 
to get quotation for each client based on their demographics.""")

### Set Sidebar Options
st.sidebar.title('About')
st.sidebar.info('Change parameters to see how insurance prices change.')
st.sidebar.title('Parameters')

# Age 
age = st.sidebar.slider('Age', 0, 100, 24)

# BMI
bmi = st.sidebar.slider('BMI', 13, 40, 31)

# Number of Children
num_children = st.sidebar.slider('Number of Children', 0, 12, 1)

# Gender
gender = st.sidebar.radio("Gender", ('female', 'male'))

if gender == 'male':
    is_female = 0
else:
    is_female = 1

# Is Smoker
smoker = st.sidebar.radio("Smoker?", ('yes', 'no'))
    
if smoker == 'yes':
    is_smoker = 1
else:
    is_smoker = 0

# Region
region = st.sidebar.selectbox("Region", ['northwest', 'northeast', 'southeast', 'southwest'])

if region == 'northeast':
    loc_list = [1, 0, 0, 0]
elif region == 'northwest':
    loc_list = [0, 1, 0, 0]
elif region == 'southeast':
    loc_list = [0, 0, 1, 0]
elif region == 'southwest':
    loc_list = [0, 0, 0, 1]


### Main Page
st.subheader('Input Data')
if st.checkbox('Show Raw Data'):
    st.write(data)

### Graphs
filters = (data['sex'] == gender) & (data['smoker'] == smoker) & (data['region'] == region)

# Distributions
if st.checkbox('Show Graphs'):
    sns.pairplot(data[filters], height=8, kind='reg', diag_kind='kde')
    st.pyplot()

### Price Output
st.subheader('Output Insurance Price')

# Model filename
filename = 'insurance_linearregression.mdl'

# load the model from disk
loaded_model = joblib.load(filename)

# [Age, BMI, Number of Children, is_female, is_smoker, is_from_NorthEast, 
prediction = round(loaded_model.predict([[age, bmi, num_children, is_female, is_smoker] + loc_list])[0])

st.write(f"Suggested Insurance Price is: {prediction}")


