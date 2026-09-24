from flask import *
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd



application=Flask(__name__)

app=application


## import ridge regression and the standard scaler
ridge_model= pickle.load(open("models/ridge.pkl","rb"))
standard_scaler= pickle.load(open("models/scaler.pkl","rb"))



@app.route("/")
def index():
    return render_template("index.html")


@app.route('/predictdata',methods=['GET' , 'POST'])
def predict_datapoint():
    if request.method=="POST":
    
        RH = float(request.form.get("RH"))
        WS = float(request.form.get("WS"))
        RAIN = float(request.form.get("RAIN"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))
        Temperature = float(request.form.get("Temperature"))

        new_data_scaled=standard_scaler.transform([[Temperature, RH, WS, RAIN, FFMC, DMC, ISI, Classes, Region]])
        result=ridge_model.predict(new_data_scaled)

        return render_template ("home.html" , result=result[0])
    
    else:
        return render_template("home.html")

if __name__=="__main__":
    app.run(debug=True , port=5000)
