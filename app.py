from flask import Flask, render_template, request
import pandas as pd
import pickle
from babel.numbers import format_currency

app = Flask(__name__)

car = pd.read_csv("Cleaned_car.csv")
model = pickle.load(open("LinearRegressionModel.pkl","rb"))

@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = sorted(car['fuel_type'].unique())

    return render_template("index.html", companies=companies, car_models=car_models, years=years,
                            fuel_types=fuel_types)


@app.route('/predict', methods = ['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_travel = int(request.form.get('kms_travel'))

    prediction = model.predict(pd.DataFrame([[car_model, company, year, kms_travel, fuel_type]],
                                             columns=['name','company','year', 'kms_driven', 'fuel_type']))

    return format_currency(prediction[0], 'INR', locale='en_IN')
