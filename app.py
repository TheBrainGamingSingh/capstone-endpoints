# > $env:FLASK_APP="app"
# > flask run

# > $env:FLASK_DEBUG=1

# python -m venv venv
# venv\Scripts\activate
import requests
import flask
from flask import Flask, render_template, url_for, make_response
## moneky patch # FIXME:
import flask.scaffold
flask.helpers._endpoint_from_view_func = flask.scaffold._endpoint_from_view_func
##

from flask_restful import reqparse, abort, Api, Resource
import pickle
import json
import re
import numpy as np
import pandas as pd

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from datetime import datetime
stemmer = PorterStemmer()
words = stopwords.words("english")

app = Flask(__name__)
api = Api(app)
parser = reqparse.RequestParser()
parser.add_argument('text_query')
parser.add_argument('district_id')
parser.add_argument('date')
parser.add_argument('cluster_data')

MODEL_PATH = './model/RandomForest.pkl'
BEARER_TOKEN = 'pec_capstone_group_12'
BASE_URL = "https://cdn-api.co-vin.in/api/v2/appointment/sessions/public/findByDistrict?district_id={}&date={}"
HEADERS = {
'origin': 'https://www.cowin.gov.in',
'referer': 'https://www.cowin.gov.in/',
'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36'
}
COLUMNS = ['vaccine','center_id', 'name','address','min_age_limit','fee_type','pincode','available_capacity_dose1','available_capacity_dose2']


with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

with open("./data/mapper.json") as infile:
    label_mapper = json.load(infile)

def clean_and_stem(text):
    return [" ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower() for x in [text]]

def get_vaccine_details(district_id, query_date):
    print('requesting using urllib')
    req_url = BASE_URL.format(district_id,query_date)
    print(req_url)
    response = requests.get(req_url,headers=HEADERS)
    try:
        response_df = pd.DataFrame(response.json()['sessions'])
        response_df = response_df[COLUMNS]
        response_df = response_df[response_df['available_capacity_dose1'] > 0]
        response_df = response_df[response_df['available_capacity_dose2'] > 0]
        response_df = response_df.sort_values(by='pincode')
        res_output = response_df.to_dict(orient='records')
        return {'details' : res_output}
    except:
        return False

# def get_vaccine_details_using_selenium(district_id,date):
#     print('requesting using selenium')
#     req_url = BASE_URL.format(district_id,date)
#     from selenium import webdriver
#     import json
#     import time
#     import os
#     GOOGLE_CHROME_PATH = os.environ['GOOGLE_CHROME_BIN']
#     CHROMEDRIVER_PATH = os.environ['GOOGLE_CHROME_SHIM']
#     chrome_options = webdriver.ChromeOptions()
#     chrome_options.add_argument('--disable-gpu')
#     chrome_options.add_argument('--no-sandbox')
#     chrome_options.binary_location = GOOGLE_CHROME_PATH
#
#     # driver = webdriver.Chrome(execution_path=CHROMEDRIVER_PATH, chrome_options=chrome_options)
#     try:
#         print('initializing webdriver...')
#         driver = webdriver.Chrome(execution_path=CHROMEDRIVER_PATH, chrome_options=chrome_options)
#         driver.get(req_url)
#         pre = driver.find_element_by_tag_name("pre").text
#         time.sleep(2)
#     except:
#         print('retrying...')
#     finally:
#         driver.quit()
#
#     try:
#         resonse_data = json.loads(pre)
#         response_df = pd.DataFrame(resonse_data['sessions'])
#         response_df = response_df[COLUMNS]
#         response_df = response_df[response_df['available_capacity_dose1'] > 0]
#         response_df = response_df[response_df['available_capacity_dose2'] > 0]
#         response_df = response_df.sort_values(by='pincode')
#         res_output = response_df.to_dict(orient='records')
#         return {'details' : res_output}
#     except:
#         return False

# end of utility fuctions














# Endpoints
class ComplaintClassifier(Resource):
    '''render a template here'''
    def get(self):
        return {"apis-available" : {
          "predict-class": {
            "method" : "GET",
            "endpoint" : "https://smart-citizen-app.herokuapp.com/api/predict-class",
            "params": "text_query"
          },

          "get-vaccine-details": {
            "method" : "GET",
            "endpoint" : "https://smart-citizen-app.herokuapp.com/api/get-vaccine-details",
            "params": ["district_id", "date"]
          },

           "get-cases-update": {
            "method" : "GET",
            "endpoint" : "https://smart-citizen-app.herokuapp.com/api/get-cases-update",
            "params": None
          },

           "get-clusters": {
            "method" : "GET",
            "endpoint" : "https://smart-citizen-app.herokuapp.com/api/get-clusters",
            "params": "cluster_data"
          }
        }}


# TODO: Put this in a try except block to report errors

class PredictClass(Resource):
    def get(self):
        args = parser.parse_args()
        print(args)

        text_query = args['text_query']

        if not text_query:
            return {'error' : 'argument text_query not found'}

        text_query = str(text_query)
        print(text_query)
        user_query = clean_and_stem(text_query)

        label = model.predict(user_query)[0]
        probs = model.predict_proba(user_query)[0]

        labels = [(i, label_mapper[str(i)], probs[i]) for i in sorted(range(len(probs)), key = lambda i: probs[i],reverse=True)][:3]
        labels_dict = []

        for i in labels:
            j = {}
            j['id'] = i[0]
            j['category'] = i[1]
            j['confidence'] = i[2]

            labels_dict.append(j)

        prediction = label_mapper[str(label)]
        confidence = int(probs[label] * 10000) / 10000

        # return the prediction, confidence and top three labels
        output = {'text_query' : text_query, 'prediction': prediction, 'confidence': confidence, 'labels' : labels_dict}
        return output

class GetVaccineDetails(Resource):
    def get(self):
        args = parser.parse_args()
        print(args)
        district_id = args['district_id']
        query_date = args['date']

        if not query_date:
            query_date = datetime.today().strftime("%d-%m-%Y")
        if not district_id:
            district_id = 108 #Chandigarh's district ID

        res = get_vaccine_details(district_id,query_date)
        if res:
            return res
        # else:
        #     res = get_vaccine_details_using_selenium(district_id,query_date)
        #     if res:
        #         return res
        else:
            return {'error' : 'No details found'}

class CasesUpdate(Resource):
    def get(self):
        url = 'https://api.covid19india.org/v4/min/timeseries.min.json'

        res = requests.get(url)
        res = res.json()['CH']['dates']
        res = res[list(sorted(res.keys()))[-1]]['total']
        res['active'] = res['confirmed'] - res['recovered'] - res['deceased']

        return {'cases' : res,
                'date' : datetime.today().strftime("%d-%m-%Y")}

class GetClusters(Resource):
    def get(self):
        args = parser.parse_args()
        print(args)
        return {'cluster_data' : 'test'}





api.add_resource(ComplaintClassifier, '/api/')
api.add_resource(PredictClass, '/api/predict-class')
api.add_resource(GetVaccineDetails, '/api/get-vaccine-details') #idk why is this not working
api.add_resource(GetClusters, '/api/get-clusters')
api.add_resource(CasesUpdate, '/api/get-cases-update')

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == '__main__':
    port = 5000
    # host = '0.0.0.0'
    app.run(debug=True,port=port)
