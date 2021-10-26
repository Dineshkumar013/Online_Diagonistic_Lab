
from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('homepage.html')
@app.route('/heart')
def heart():
    return render_template('heart_pred.html')
@app.route('/diabetes')
def diabetes():
    return render_template('diabetes_pred.html')
@app.route('/parkinson')
def parkinson():
    return render_template('parkinson_pred.html')
@app.route('/thyroid')
def thyroid():
    return render_template('thyroid_pred.html')
@app.route('/kidney')
def kidney():
    return render_template('kidney_pred.html')
@app.route('/liver')
def liver():
    return render_template('liver_pred.html')
@app.route('/h_predict', methods=['POST'])
def heart_predict():
    model=pickle.load(open('E:\MINI project\model_files\heart-new.pkl','rb')) 
    name=request.form.get("n")
    initial=[float(x) for x in request.form.values()]
    arr =[np.array(initial)]
    pred = model.predict(arr)
    if pred == 0:
        
        return render_template('heart_pred.html',prediction_text="Hurray !  Your body results doesnot have the symptoms of Heart disease...")
    elif pred == 1:
        return render_template('heart_pred.html',prediction_text="Sorry to say this ...  you are having the symptoms of heart disease consult a docter immediately...")

@app.route('/db_predict',methods=['POST'])
def diabetes_predict():  
    model=pickle.load(open(r'E:\MINI project\model_files\new_diabetes.pkl','rb')) 
    
    initial=[float(x) for x in request.form.values()]
    arr =[np.array(initial)]
    pred = model.predict(arr)
    if pred == 0:
        
        return render_template('diabetes_pred.html',prediction_text="Hurray !  Your body results doesnot have the symptoms of any diabetes disease...")
    elif pred == 1:
        return render_template('diabetes_pred.html',prediction_text="Sorry to say this ... you are having the symptoms of diabetes disease consult a docter immediately...")

@app.route('/p_predict',methods=['POST'])
def parkinson_predict():  
    model=pickle.load(open(r'E:\MINI project\model_files\parkinson.pkl','rb')) 
    
    initial=[float(x) for x in request.form.values()]
    arr =[np.array(initial)]
    pred = model.predict(arr)
    if pred == 0:
        
        return render_template('parkinson_pred.html',prediction_text="Hurray !  Your body results doesnot have the symptoms of parkinson disease...")
    elif pred == 1:
        return render_template('parkinson_pred.html',prediction_text="Sorry to say this ... you are having the symptoms of parkinson disease consult a docter immediately...")
@app.route('/t_predict',methods=['POST'])
def thyroid_predict():  
    model=pickle.load(open(r'E:\MINI project\model_files\Thyroid.pkl','rb')) 
    
    initial=[float(x) for x in request.form.values()]
    arr =[np.array(initial)]
    pred = model.predict(arr)
    if pred == 0:
        
        return render_template('thyroid_pred.html',prediction_text="Hurray !  Your body results doesnot have the symptoms of thyroid disease...")
    elif pred == 1:
        return render_template('thyroid_pred.html',prediction_text="Sorry to say this ... you are having the symptoms of parkinson thyroid consult a docter immediately...")

@app.route('/k_predict',methods=['POST'])
def kidney_predict():  
    model=pickle.load(open(r'E:\MINI project\model_files\kidney.pkl','rb')) 
    
    initial=[float(x) for x in request.form.values()]
    arr =[np.array(initial)]
    pred = model.predict(arr)
    if pred == 0:
        
        return render_template('kidney_pred.html',prediction_text="Hurray !  Your body results doesnot have the symptoms of kidney disease...")
    elif pred == 1:
        return render_template('kidney_pred.html',prediction_text="Sorry to say this ... you are having the symptoms of kidney disease consult a docter immediately...")

@app.route('/l_predict',methods=['POST'])
def liver_predict():  
    model=pickle.load(open(r'E:\MINI project\model_files\LIVER.pkl','rb')) 
    
    initial=[float(x) for x in request.form.values()]
    arr =[np.array(initial)]
    pred = model.predict(arr)
    if pred == 0:
        
        return render_template('liver_pred.html',prediction_text="Hurray !  Your body results doesnot have the symptoms of liver disease...")
    elif pred == 1:
        return render_template('liver_pred.html',prediction_text="Sorry to say this ... you are having the symptoms of liver disease consult a docter immediately...")


if __name__ == "__main__":
    app.run(debug=True)


