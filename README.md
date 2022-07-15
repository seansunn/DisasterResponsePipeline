### Disaster Response Pipeline Project

## Project descripton
This project creates a web app to categorize messages related to disaster. The data from Figure Eight is preprocessed through ETF pipeline. Then, model is trained through ML pipeline on the cleaned data and dumped through pickle. Finally, flask and plotly is used to deploy the model.

## Instructions:
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
Thanks to Figure Eight for providing the data. Feel free to use any of the code in the jupyter notebook.
