# Disaster Response Pipeline Project
Communication is the key to coordination of efforts when a disaster strikes. To have a clear vision in
such a situation, we need to extract useful information from the incoming messages. Otherwise we can easily
be overwhelmed by the amounts of data coming in. This tool can help in the process. Trained on over 25 thousand preclassified images, the resulting model allows to categorize any new incoming text to a combination of 35 categories.

```
'related', 'request', 'offer', 'aid_related', 'medical_help',
'medical_products', 'search_and_rescue', 'security', 'military',
'water', 'food', 'shelter', 'clothing', 'money', 'missing_people',
'refugees', 'death', 'other_aid', 'infrastructure_related', 'transport',
'buildings', 'electricity', 'tools', 'hospitals', 'shops',
'aid_centers', 'other_infrastructure', 'weather_related', 'floods',
'storm', 'fire', 'earthquake', 'cold', 'other_weather',
'direct_report'
```

### Test deployment
Try out the app [here](https://disaster-response-pipelines.herokuapp.com/). Note that when the app hasn't been used for an hour, Heroku will put it to sleep, so it takes up to 30s before it is usable. Subsequent reloads and usage perform normally.

### Libraries used
See the `requirements.txt` file.

### Deployment Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Acknowledgements
Thanks to Udacity for providing a skeleton of the application, which has made much easier to finish the project.
