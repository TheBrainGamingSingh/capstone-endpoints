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
parser.add_argument('query')
parser.add_argument('district_id')
parser.add_argument('date')
parser.add_argument('cluster_data')

MODEL_PATH = './model/RandomForest.pkl'
BEARER_TOKEN = 'pec_capstone_group_12'
BASE_URL = "https://cdn-api.co-vin.in/api/v2/appointment/sessions/public/findByDistrict?district_id={}&date={}"
HEADERS = {
'authorization': 'Bearer {}'.format(BEARER_TOKEN),
'authority': 'cdn-api.co-vin.in',
'scheme' : 'https',
'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
'accept-encoding': 'gzip, deflate, br',
'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36'
}
COLUMNS = ['vaccine','center_id', 'name','address','min_age_limit','pincode','available_capacity_dose1','available_capacity_dose2']


with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

with open("./data/mapper.json") as infile:
    label_mapper = json.load(infile)

def clean_and_stem(text):
    return [" ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower() for x in [text]]


def get_vaccine_details(district_id, query_date):
    req_url = BASE_URL.format(district_id,query_date)
    print(req_url)
    response = requests.get(req_url,headers=HEADERS)
    if response:
        response_df = pd.DataFrame(response.json()['sessions'])
        response_df = response_df[COLUMNS]
        return response_df.to_json(orient="split")

    return {'error' : 'No details found',
            'response': str(response)}
# Endpoints
class ComplaintClassifier(Resource):
    def get(self):
        return {'Welcome!': '''
        This is the API endpoint of our capstone project, Smart Citizen App!

        Available APIs:
        POST    https://capstone-classifier.herokuapp.com/predict  Parameters: text_query
        GET     https://capstone-classifier.herokuapp.com/get-vaccine-details Parameters: district_id
        GET     https://capstone-classifier.herokuapp.com/get-clusters
        GET     https://capstone-classifier.herokuapp.com/cases-update


        '''}

# TODO: Put this in a try except block to report errors

class PredictClass(Resource):
    def post(self):
        args = parser.parse_args()
        print(args)
        text_query = str(args['query'])
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

        return get_vaccine_details(district_id,query_date)

class GetClusters(Resource):
    def get(self):
        args = parser.parse_args()
        print(args)
        return {'cluster_data' : 'test'}

class CasesUpdate(Resource):
    def get(self):
        url = 'https://api.covid19india.org/v4/min/timeseries.min.json'

        res = requests.get(url)
        res = res.json()['CH']['dates']
        res = res[list(sorted(res.keys()))[-1]]['total']
        res['active'] = res['confirmed'] - res['recovered'] - res['deceased']

        return {'cases' : res,
                'date' : datetime.today().strftime("%d-%m-%Y")}

api.add_resource(ComplaintClassifier, '/api')
api.add_resource(PredictClass, '/predict')
api.add_resource(GetVaccineDetails, '/get-vaccine-details') #idk why is this not working
api.add_resource(GetClusters, '/get-clusters')
api.add_resource(CasesUpdate, '/cases-update')

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == '__main__':
    port = 5000
    app.run(debug=True, port=port)
