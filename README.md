# Disaster Response Pipeline Project


## Project descripton

This project creates a web app to categorize messages related to disaster. The data from Figure Eight is preprocessed through ETF pipeline. Then, model is trained through ML pipeline on the cleaned data and dumped through pickle. Finally, flask and plotly is used to deploy the model.

The web app is potentially helpful to the community in an event of disaster for it's ability to classify the messages and the related departments can use the information to efficiently act on people's needs in the disaster.

## File description

app\
| - template\
| |- master.html `main page of web app`\
| |- go.html `classification result page of web app`\
|- run.py `flask file that runs app`\
data\
|- disaster_categories.csv `raw data to process`\
|- disaster_messages.csv `raw data to process`\
|- process_data.py `script to process the raw data and then save as sql database`\
|- DisasterResponse.db `database to save clean data to`\
models\
|- train_classifier.py `script to train the xgboost model on clean data and dump the model`\
|- classifier.pkl `saved xgboost model`\
README.md

## Instructions

0. Install xgboost for running this app.

1. Run the following commands in the project's root directory to set up database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Licensing, Acknowledgements
Thanks to Figure Eight for providing the data. Feel free to use any of the code.
