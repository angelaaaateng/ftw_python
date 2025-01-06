# Import Packages
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
import joblib 
from sklearn.linear_model import LinearRegression 

# Fast API
from fastapi import FastAPI

app = FastAPI()


# Load model using joblib
saved_model = joblib.load('adsales_linearregression.mdl')


@app.get("/")
def read_root():
    return {"Hello": "World"}

# assign default costs
# tv_cost = 150
# radio_cost = 25
# news_cost = 125

@app.get("/predict_sales/")
def read_item(tv_cost:int=150, radio_cost:int= 25,news_cost:int=125):
    predicted_sales = saved_model.predict([[tv_cost, radio_cost, news_cost]])[0]
    return {"predict_sales": "Predicted sales is "+str(predicted_sales*1000)+" dollars."}



##python -m uvicorn main:app --reload