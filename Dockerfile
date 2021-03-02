FROM python:3.9.2-slim-buster

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

RUN python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
RUN python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

CMD python ./app/run.py --port "$PORT"