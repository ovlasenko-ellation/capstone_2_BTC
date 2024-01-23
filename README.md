# Capstone-2 Project price change trend for trading data

##Summary:
This project will include the following:

1. Problem Description
2. EDA and Dataset preparation
3. Model Training, tuning and evaluation
4. Installation steps
5. Running predictions locally
6. Model deployment and test with Flask and Containerization
7. AWS Ddeployment
   1. Upload docker image to ECR
   2. Create Lambda function
   3. Create API Gateway service
8. AWS deploy The idea is to apply everything learned so far.

##Problem Description
Primary goal of the project is the ability to predict the price trend change for trading data based on the trading dataset and calculated indexes and features. This would be helpful in predicting whether to buy or sell items in portfolio basing on the market situation and upcoming trends.

Dataset is the daily market data for trades. The full dataset is available on [Kaggle](https://www.kaggle.com/datasets/spoorthiuk/crypto-market-data-2023?resource=download)
Also a dataset from any open source can be used, such as [Yahoo](https://finance.yahoo.com/quote/BTC-USD/history?p=BTC-USD)

##EDA and Dataset preparation
For preparing data to analysis I did the following:
1. Transformed columns into the lowercase and replaced spaces with underscored, checked for missing values
2. Added new features for calculating the RSI index and price difference with previous and upcoming day, replace NaN values with 0
3. Added visual distribution of data
4. Checked feature correlation for numeric values
5. Once all the analysis and data preparation were done I split the dataset into full_train and test, and then split full_train into train and validation datasets. Used a recommended split by period for the timeseries data

##Model Training, tuning and evaluation
Since the goal for model is to predict whther the price will go up or down I used models for binary classification:
- LogisticRegression
- RandomForest
- XGBoost

For validation purposes used accuracy, F1 score (balanced between preciosn and recall) and auc metrics.
All models include parameter tuning and selection based on the optimal f1/auc/accuracy scores
Since dataset is relatively small used KFold cross-validation for all models. 
XGBoost showed better perfromance on the full_train dataset, thus it was selected as a main model.

##Installation steps
1. Clone the repository
```
$ git clone git@github.com:ovlasenko-ellation/capstone_2_BTC.git
```
2. Active venv virtual environemnt
```
python3 -m venv venv`
source venv/bin/activate
```
3. Install required dependencies
`$ pip install -r requirements.txt`

##To run project locally
1. Run predict_lambda.py to launch the service  on a local server
`python predict_lambda.py`
2. In the input.py uncomment the local host url and comment the docket one and run in new terminal window 
`python input.py`

##Model deployment and test with Flask

Containerzation can be done to test the model with the flask locally. 
Here are the steps:
1. Rename Dockerfile_local to a Dockerfile (current version of Dockerfile contains lambda configuration
2. Run the command to build docker 
```
docker build . -t "btc_price_trend_predictor"
```
3. Run the docker 
```
docker run -p 8080:9696 btc_price_trend_predictor :latest
```
4. Update the url in input.py to the docker url (uncomment if commented and comment the others)
5. In new terminal window run 
`python input.py`

##AWS Deployment

##Creating Docker image for ECR
1. Use the original Dockerfile from the project
2. Run the command to build docker 
```
docker build . -t "btc_price_trend_predictor"
```
3. Create a new repository in Amazon ECR via the AWS Management Console.
```
$ aws ecr create-repository --repository-name <repository-name>
```
Don't forget to copy repositoryUri value to be able to continue.
4. Authenticate your Docker client to the Amazon ECR registry you just created.
```
$(aws ecr get-login --no-include-email)
```
5. Tag the docker image with repositoryUri
```
docker tag btc_price_trend_predictor:latest <repositoryUri>:btc_prediction
```
6. Push the image to ECR
```docker push <repositoryUri>:btc_prediction_xgb_model```
7. Log in to AWS colsole -> Amazon ECR and you will be able to see it added
![ECR](https://github.com/ovlasenko-ellation/capstone_2_BTC/blob/main/Images/ECR.png)

##Create AWS Lambda function
1. Go to the Lambda service in the AWS Management Console and choose 'Create function'.
[Lambda Create](https://github.com/ovlasenko-ellation/capstone_2_BTC/blob/main/Images/lambda_create.png)
2. Select the 'Container image' option, provide a name for your Lambda function and choose the Docker image you uploaded to ECR as the container image.
3. If there is an error, configure any additional settings such as memory (1024), timeout (1 minute) :
4. Test the function once created by selecting the `Test` tab and providing the following imput
```
{
  "date": "2023-12-27",
  "open": 42518.46875,
  "high": 43683.160156,
  "low": 42167.582031,
  "close": 43442.855469,
  "adj_close": 43442.855469,
  "volume": 25260941032,
  "price_change": 922.453125,
  "rsi": 52.953577,
  "is_bull": "true",
  "overbought": "false",
  "oversold": "false"
}
```
[Lambda Test](https://github.com/ovlasenko-ellation/capstone_2_BTC/blob/main/Images/lambda_test.png)

##Create API Gateway Service
1. Navigate to API Gateway service, then click on Create, if this is your first API you will be reditected to nex step Choose an API type:
[Add API](https://github.com/ovlasenko-ellation/capstone_2_BTC/blob/main/Images/create_gatewayAPI.png)
2. Select `REST API` -> `Build` -> `New API`. Add API name and press `Create API`
3. Once API is created, add a new resource by clicking on Create resource `Create Resource`
4. In resource details specify `predict` in resource name
5. Once endpoint is added select it and press `Create Method`
[Add Methd](https://github.com/ovlasenko-ellation/capstone_2_BTC/blob/main/Images/create_method.png)
6. Select from dropdown Method Type - POST, integration type - Lambda function and press `Create Method`
7. Select newly created POST method and go to the `Test` tab
8. Add the following in the `Request Body` section and press test 
```
{
  "date": "2023-12-27",
  "open": 42518.46875,
  "high": 43683.160156,
  "low": 42167.582031,
  "close": 43442.855469,
  "adj_close": 43442.855469,
  "volume": 25260941032,
  "price_change": 922.453125,
  "rsi": 52.953577,
  "is_bull": "true",
  "overbought": "false",
  "oversold": "false"
}
```
[API Gateway Test](images/API_TEST.png)