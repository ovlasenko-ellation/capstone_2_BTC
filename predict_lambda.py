import json
import pickle
import pandas as pd
import xgboost as xgb
from flask import Flask, request, jsonify

model_file = 'model_new.bin'
dv_file = 'dict_vectorizer.pickle'

def load_model(model_file):
    with open(model_file, 'rb') as file:
        model = pickle.load(file)
    return model

def load_dict_vectorizer(dv_file):
    with open(dv_file, 'rb') as file:
        dv = pickle.load(file)
    return dv

dv = load_dict_vectorizer(dv_file)
model = load_model(model_file)

def predict(input_data):
    # Transform input data into a DataFrame
    input_df = pd.DataFrame([input_data])

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
        'tomorrow_price_growing_prediction': bool(binary_predictions[0])
    }

    return result

app = Flask('get-price-trend-prediction')

@app.route('/predict', methods=['POST'])
def flask_predict():
    input_data = request.get_json()
    print(f"Request received: {input_data}")
    result = predict(input_data)
    return jsonify(result)

def lambda_handler(event, context):
    #input_data = json.loads(event)
    result = predict(event)
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
