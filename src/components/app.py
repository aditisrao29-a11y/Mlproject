from flask import Flask,request,render_template
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import Customdata,Predictpipeline
application=Flask(__name__)

app=application
@app.route('/')
def index():
    return render_template('home.html')
@app.route('/predicted',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=Customdata(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            math_score=request.form.get('math_score')
        )
        pred_df=data.get_data_as_df()
        print(pred_df)
        predict_pipeline=Predictpipeline()
        result=predict_pipeline.predict(pred_df)
        return render_template('home.html',result=result[0])
    
if __name__=='__main__':
    app.run(host='0.0.0.0',debug=True)
