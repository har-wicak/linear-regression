from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def weight_prediction():
    if request.method == 'GET':
        return render_template("weight-prediction.html")
    elif request.method == 'POST':
        print(dict(request.form))
        weight_features = dict(request.form).values()
        weight_features = np.array([float(x) for x in weight_features])
        print(weight_features)
        lin_regr_model, std_scaler_height, std_scaler_weight = joblib.load("model-dev/height-to-weight-prediction-linear-regression.pkl")
        weight_features[1:2]= std_scaler_height.transform(weight_features[1:2].reshape(1, -1))
        weight_features = np.reshape(weight_features, (1, 2))
        print('weight: ' + str(weight_features))
        result = lin_regr_model.predict(weight_features)
        print(result)
        result = std_scaler_weight.inverse_transform(result.reshape(1, -1))
        result = np.array2string(result)
        result = result[2:-2]
        print('final results' + str(result))
        return render_template('weight-prediction.html', result=result)
    else:
        return "Unsupported Request Method"


if __name__ == '__main__':
    app.run(port=5000, debug=True)