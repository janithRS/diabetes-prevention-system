import json
import pickle
from json import JSONEncoder

from flask import Flask, request, url_for, redirect, render_template, jsonify
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def get_index():
    print("hello")
    return render_template("diabetes_predict.html")


# @app.route('/stress')
# def get_stress_questionnaire():
#     return render_template("stress_questionnaire.html")
#
#
# @app.route('/test_ionic')
# def test_ionic():
#     return render_template('predict_diabetes_form.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = [float(x) for x in request.args.values()]
    final = [np.array(int_features)]
    prediction = model.predict(final)
    prediction_prob = model.predict_proba(final)
    new_output = {
        "prediction_prob_of_positive": (prediction_prob[0][1]*100).item(),
        "prediction_prob_of_negative": (prediction_prob[0][0]*100).item(),
        "prediction": (prediction[0]).item()
        }
    output_json = json.dumps(new_output)

    return output_json
    # return render_template('diabetes_result.html', predictValue=float(prediction))


if __name__ == '__main__':
    app.run(debug=True)
