import pickle

from flask import Flask, request, url_for, redirect, render_template, jsonify
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def hello_world():
    return render_template("diabetes_predict.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    print(request)
    int_features = [float(x) for x in request.form.values()]
    final = [np.array(int_features)]
    # print(int_features)
    # print(final)
    prediction = model.predict_proba(final)
    output = '{0:.{1}f}'.format(prediction[0][1], 2)

    # return jsonify(output);
    return render_template('diabetes_predict.html', predictValue=float(output))


if __name__ == '__main__':
    app.run()
