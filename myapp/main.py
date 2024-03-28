import numpy as np
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
from myapp.model import Model
from warnings import filterwarnings
filterwarnings('ignore')

# load model
model = Model()

class Form(BaseModel):
    index: int
    session_id: str
    client_id: str
    visit_date: str
    visit_time: str
    visit_number: int
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_model: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str

# app
app = FastAPI()

# api
@app.post('/predict')
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    pred = int(model.predict(df)[0])
    return {'Prediction': pred}

@app.get('/status')
def status():
    return "I'm alive"