

from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

pickle_in = open("model.pkl","rb")
classifier=pickle.load(pickle_in)



@app.route('/')
def welcome():
    return "Predictive-Tests-For-Assessing-Risk-of-Cancer-Recurrence"

@app.route('/predict',methods=["Get"])
def predict_note_authentication():
    
    """Predictive-Tests-For-Assessing-Risk-of-Cancer-Recurrence
    Predictive-Tests-For-Assessing-Risk-of-Cancer-Recurrence.
    ---
    parameters:  
      - name: Tumor_size
        in: query
        type: number
        required: true
      - name: Node_Status_of_the_Tumor
        in: query
        type: number
        required: true
      - name: Age
        in: query
        type: number
        required: true
      - name: Tumor_Grade
        in: query
        type: number
        required: true
      - name: Anomaly_score
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    Tumor_size=request.args.get("Tumor_size")
    Node_Status_of_the_Tumor=request.args.get("Node_Status_of_the_Tumor")
    Age=request.args.get("Age")
    Tumor_Grade=request.args.get("Tumor_Grade")
    Anomaly_score=request.args.get("Anomaly_score")
    prediction=classifier.predict([[Tumor_size,Node_Status_of_the_Tumor,Age,Tumor_Grade,Anomaly_score]])
    print(prediction)
    return "[One Repersent High chance of Cancer Recurrence & Zero Repersent  Low chance of Cancer Recurrence] Your Result  :--"+str(prediction)

@app.route('/predict_file',methods=["POST"])
def predict_note_file():
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """
    df_test=pd.read_csv(request.files.get("file"))
    print(df_test.head())
    prediction=classifier.predict(df_test)
    
    return str(list(prediction))

if __name__=='__main__':
    app.run()
    
    