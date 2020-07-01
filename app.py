import json
import pickle

from flask import Flask, request, url_for, redirect, render_template, jsonify
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def get_index():
    print("hello")
    return render_template("diabetes_predict.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    # print(request.form)
    int_features = [float(x) for x in request.form.values()]
    final = [np.array(int_features)]
    prediction = model.predict(final)
    prediction_prob = model.predict_proba(final)
    output = np.append(prediction_prob, prediction[0])
    # print(output)
    output_json = json.dumps(output.tolist())
    # print(type(prediction_json))
    # output = '{0:.{1}f}'.format(prediction[0][1], 2)

    return output_json
    # return render_template('diabetes_result.html', predictValue=float(prediction))


if __name__ == '__main__':
    app.run(debug=True)
