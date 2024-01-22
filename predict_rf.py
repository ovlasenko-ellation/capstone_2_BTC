import pickle

from flask import Flask
from flask import request
from flask import jsonify


"""##Load model"""

model_file = 'model.bin'

def load(model_file: str):
    with open(model_file, 'rb') as f_in:
        dv, rf = pickle.load(f_in)
        return dv, rf

dv, rf = load(model_file)


app = Flask('get-price-trend-prediction')


@app.route('/predict', methods=['POST'])
def predict():
    input = request.get_json()
    print(f"request received {request.get_json()}")
    x = dv.transform([input])
    y = rf.predict(x)

    #value_mapping = {0: 'price decreasing', 1: 'price increasing'}
    #price_prediction = [value_mapping[value] for value in y]


    result = {
        'tomorrow_price_growing_prediction': bool(y)
    }
    return jsonify(result)

if __name__ == "__main__":

    app.run(debug=True, host='0.0.0.0', port=9696)

