from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

model=pickle.load(open("models/linreg.pkl","rb"))
scalar=pickle.load(open("models/scalar.pkl","rb"))


application=Flask(__name__)
app=application

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata",methods=['GET','POST'])
def predict_datapoint():
    if request.method=="POST":
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        WS=float(request.form.get('WS'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))

        new_data=scalar.transform([[Temperature,RH,WS,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=model.predict(new_data)

        return render_template('home.html',results=result[0])
    else:
        return render_template('home.html')
def index():
    return render_template("index.html")

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8080)