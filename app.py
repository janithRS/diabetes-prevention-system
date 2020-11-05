import json
import pickle
import sys
from flask_cors import CORS
from flask import Flask, request, render_template
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


# @app.route('/')
# def get_index():
#     print("hello")
#     print(sys.path)
#     return render_template("diabetes_predict.html")


@app.route('/getQuiz')
def get_stress_questionnaire():
    with open('quiz.json') as json_file:
        data = json.load(json_file)
    output_json = json.dumps(data)
    return output_json


@app.route('/getLifestyleTherapy')
def get_lifestyle_therapy():
    with open('lifestyle_therapy.json') as json_file:
        data = json.load(json_file)
    output_json = json.dumps(data)
    return output_json


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    data = request.data
    print(data)
    datadict = json.loads(data)
    print(datadict)

    preg_arg = int(datadict['preg'])
    glu_arg = int(datadict['glu'])
    bp_arg = int(datadict['bp'])
    skin_arg = int(datadict['skin'])
    ins_arg = int(datadict['ins'])
    bmi_arg = int(datadict['bmi'])
    ped_arg = int(datadict['ped'])
    age_arg = int(datadict['age'])

    result = analyze_bmi(float(bmi_arg))
    ob_stage, ob_therapy = result

    int_features = [preg_arg, glu_arg, bp_arg, skin_arg, ins_arg, bmi_arg, ped_arg, age_arg]
    final = [np.array(int_features)]

    prediction = model.predict(final)
    prediction_prob = model.predict_proba(final)
    prediction_output = 'Positive' if (prediction[0]).item() == 1 else 'Negative'

    rec_res = give_rec(ob_stage)
    nutrition_info, physical_info, sleep_info, behavioral_info, smoking_info = rec_res

    output_result = {
        "Message": "The result predictions according to the inputs",
        "dm_prediction_prob_of_positive": int((prediction_prob[0][1] * 100).item()),
        "dm_prediction_prob_of_negative": int((prediction_prob[0][0] * 100).item()),
        "dm_prediction": prediction_output,
        "ob_stage": ob_stage,
        "ob_therapy": ob_therapy,
        "nutrition_info": nutrition_info,
        "physical_info": physical_info,
        "sleep_info": sleep_info,
        "behavioral_info": behavioral_info,
        "smoking_info": smoking_info
    }

    output_json = json.dumps(output_result)
    return output_json

def give_rec(ob_stage):
    lst = get_lifestyle_therapy()
    d = json.loads(lst)
    nutrition_info = 'No recommendations'
    physical_info = 'No recommendations'
    sleep_info = 'No recommendations'
    behavioral_info = 'No recommendations'
    smoking_info = 'No recommendations'

    if ob_stage == 1:
        nutrition_info = d['Nutrition'][0]['con']
        physical_info = d['Physical'][0]['con']
        sleep_info = d['Sleep'][0]['con']
        behavioral_info = d['Behavioral'][0]['con']
        smoking_info = d['Smoking'][0]['con']
    elif ob_stage == 2:
        nutrition_info = d['Nutrition'][1]['con']
        physical_info = d['Physical'][1]['con']
        sleep_info = d['Sleep'][1]['con']
        behavioral_info = d['Behavioral'][1]['con']
        smoking_info = d['Smoking'][1]['con']
    elif ob_stage == 3:
        nutrition_info = d['Nutrition'][2]['con']
        physical_info = d['Physical'][2]['con']
        sleep_info = d['Sleep'][2]['con']
        behavioral_info = d['Behavioral'][2]['con']
        smoking_info = d['Smoking'][2]['con']

    return nutrition_info, physical_info, sleep_info, behavioral_info, smoking_info


def analyze_bmi(bmi):
    ob_stage = 0
    therapy = 'None'
    if 25.0 <= bmi <= 29.9:
        ob_stage = 0
        therapy = 'None'
    if 30.0 <= bmi <= 34.9:
        ob_stage = 1
        therapy = 'Lifestyle'
    if 35.0 <= bmi <= 39.9:
        ob_stage = 2
        therapy = 'Medical'
    if bmi > 40.0:
        ob_stage = 3
        therapy = 'Surgical'
    return ob_stage, therapy


if __name__ == '__main__':
    app.run(debug=True)
