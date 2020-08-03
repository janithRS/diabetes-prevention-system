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


@app.route('/')
def get_index():
    print("hello")
    print(sys.path)
    return render_template("diabetes_predict.html")


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
    preg_arg = request.args.get('preg')
    glu_arg = request.args.get('glu')
    bp_arg = request.args.get('bp')
    skin_arg = request.args.get('skin')
    ins_arg = request.args.get('ins')
    bmi_arg = request.args.get('bmi')
    ped_arg = request.args.get('ped')
    age_arg = request.args.get('age')

    result = analyze_bmi(float(bmi_arg))
    ob_stage, ob_therapy = result

    int_features = [preg_arg, glu_arg, bp_arg, skin_arg, ins_arg, bmi_arg, ped_arg, age_arg]
    final = [np.array(int_features)]

    prediction = model.predict(final)
    prediction_prob = model.predict_proba(final)
    prediction_output = 'Positive' if (prediction[0]).item() == 1 else 'Negative'
    output_result = {
        "Message": "The result predictions according to the inputs",
        "dm_prediction_prob_of_positive": int((prediction_prob[0][1] * 100).item()),
        "dm_prediction_prob_of_negative": int((prediction_prob[0][0] * 100).item()),
        "dm_prediction": prediction_output,
        "ob_stage": ob_stage,
        "ob_therapy": ob_therapy
    }

    output_json = json.dumps(output_result)
    return output_json


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
