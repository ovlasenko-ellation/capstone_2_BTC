FROM public.ecr.aws/lambda/python:3.12

# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install the specified packages
RUN pip install -r requirements.txt

# Copy function code
COPY predict_lambda.py ${LAMBDA_TASK_ROOT}
COPY model_new.bin ${LAMBDA_TASK_ROOT}
COPY dict_vectorizer.pickle ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "predict_lambda.lambda_handler" ]
