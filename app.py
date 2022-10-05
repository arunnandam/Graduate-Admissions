# Importing the packages
import pickle
from flask import Flask, request, app, jsonify, render_template, url_for
import numpy as np
import pandas as pd

# Starting point of Application
app = Flask(__name__)

# Create home page
@app.route('/')
def home():
    return render_template('home.html')

# Read scaler and model to transform and give prediction
scaler = pickle.load(open('scaling.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Give Predictions using Postman(API testing application)
@app.route('/predict',methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    #print(data)
    transformed_data = scaler.transform(np.array(list(data)).reshape(1,-1))
    output = model.predict(transformed_data)
    output = "Admit chance: " + str(round(output[0]*100,3)) + '%'
    return render_template('home.html', prediction_text = "The house price for the {}".format(output))

# App run start here. Initialize the app
if __name__=="__main__":
    app.run(debug=True)


