import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application


## import xgboost and standard scaler pickle

xgboost_model=pickle.load(open('Models/xgboost_model.pkl','rb'))
standard_scaler=pickle.load(open('Models/standard_scaler.pkl','rb'))

@app.route('/predictdata',methods=['GET','POST'])
def predict_radiation():
    result = None
    if request.method == 'POST':
        # Retrieve form data
        temperature = float(request.form['temperature'])
        pressure = float(request.form['pressure'])
        humidity = float(request.form['humidity'])
        windDirection = float(request.form['windDirection'])
        speed = float(request.form['speed'])
        month = int(request.form['month'])
        day = int(request.form['day'])
        hour = int(request.form['hour'])
        minute = int(request.form['minute'])
        second = int(request.form['second'])
        risehour = int(request.form['risehour'])
        riseminute = int(request.form['riseminute'])
        sethour = int(request.form['sethour'])
        setminute = int(request.form['setminute'])

        new_scaled_data=standard_scaler.transform([[temperature,pressure,humidity,windDirection,speed,month,day,hour,minute,second,risehour,riseminute,sethour,setminute]])
        result=xgboost_model.predict(new_scaled_data)

        return render_template('index.html',results=result[0])
    else:
        return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)