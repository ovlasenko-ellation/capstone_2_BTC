FROM python:3.11.5-slim

WORKDIR /app
COPY ["requirements.txt", "./"]
RUN pip install -r requirements.txt

COPY ["model.bin", "./"]

COPY ["predict.py", "./"]
EXPOSE 9696
ENTRYPOINT ["gunicorn", "-b", "0.0.0.0:9696", "predict:app"]