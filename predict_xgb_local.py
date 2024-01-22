import pickle
import pandas as pd
import xgboost as xgb

from flask import Flask
from flask import request
from flask import jsonify

"""##Load model"""
model_file = 'model_new.bin'
dv_file = 'dict_vectorizer.pickle'


def load(model_file: str):
    # Load the final model from the file
    with open(model_file, 'rb') as model_file:
        model = pickle.load(model_file)
        return model


def load_dict_vectorizer(dv_file: str):
    with open(dv_file, 'rb') as dv_file:
        dv = pickle.load(dv_file)
    return dv


dv = load_dict_vectorizer(dv_file)
model = load(model_file)

app = Flask('get-price-trend-prediction')


@app.route('/predict', methods=['POST'])
def predict():
    input = request.get_json()
    print(f"request received {request.get_json()}")

    # Transform input data into a DataFrame
    input_df = pd.DataFrame([input])

    # Use the loaded DictVectorizer to transform the input data
    X_input = dv.transform(input_df.to_dict(orient='records'))

    # Convert feature names to a list
    feature_names_list = dv.get_feature_names_out().tolist()

    # Create an XGBoost DMatrix
    dinput = xgb.DMatrix(X_input, feature_names=feature_names_list)

    # Use the loaded XGBoost model to make predictions
    predictions = model.predict(dinput)

    # Convert probabilities to binary predictions (adjust the threshold if needed)
    binary_predictions = (predictions > 0.5).astype(int)

    # Print the predictions
    print("Predictions:", binary_predictions)

    result = {
        'tomorrow_price_growing_prediction': bool(binary_predictions)
    }
    return jsonify(result)

# def lambda_handler(event, context):
#     url = event['url']
#     predictions = model.predict(dinput)
#     binary_predictions = (predictions > 0.5).astype(int)
#     result = {
#         'tomorrow_price_growing_prediction': bool(binary_predictions)
#     }
#     return result

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
