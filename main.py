from fastapi import FastAPI 
from pydantic import BaseModel
import uvicorn 
import numpy as np 
import pandas as pd 
from fastapi.middleware.cors import CORSMiddleware
from prediction_model.predict import generate_pred

app = FastAPI(
    title='Loan Prediction App w/ Jenkins',
    description='CI CD with Jenkins',
    version='1.0'
)

origins = [
    '*'
]

# Adding middleware for more verbose RESTful api
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

class LoanPrediction(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float
    Property_Area: str

@app.get('/')
def index():
    return {'message': 'Welcome to Loan Prediction App w/ Jenkins'}

@app.post('/predict')
def predict(loan_details: LoanPrediction):
    data = loan_details.model_dump() # Turn json object into python dictionary
    prediction = generate_pred([data])['Predictions'][0]
    if prediction == 'Y':
        pred = 'Approved'
    
    else: 
        pred = 'Rejected'
    return {'status':pred}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8005)

