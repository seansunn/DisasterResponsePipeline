import re
import json
import plotly
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    '''
    StartingVerbExtractor
    transform messages by tagging and extracting the starting verb
    
    Input:
    BaseEstimator       sklearn BaseEstimator function 
    TransformerMixin    sklearn TransformerMixin function 
    
    Functions:
    starting_verb       text --> 0 or 1
    fit                 returns self
    transform           applys transformation on text and returns dataframe
    '''
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            if pos_tags:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return 1
        return 0
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def tokenize(text):
    '''
    tokenize
    transfer the raw text message into cleaned tokens
    
    Input:
    text        raw text message
    
    Returns:
    clean_tokens    processed cleaned tokens
    '''
    # remove punctuation and make all characters in lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize words
    tokens = word_tokenize(text)
    
    # remove possible stop words and empty strings
    tokens = [w.strip() for w in tokens if w not in stopwords.words("english") if w]
    
    # initiate a wordnet lemmatizer object
    lem = WordNetLemmatizer()
    
    # lemmatize every word
    clean_tokens = [lem.lemmatize(tok, pos='v') for tok in [lem.lemmatize(t) for t in tokens]]

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)


# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # extract top 10 most frequent categories
    top_cat = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    top_cat_lst = top_cat.sum().sort_values(ascending=False)
    top_names = top_cat_lst.index.tolist()[:10]
    top_values = top_cat_lst.values.tolist()[:10]

    # create visuals
    graphs = [
        {
            # data and layouts for graph 1
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        {
            # data and layouts for graph 2
            'data': [
                Bar(
                    x=top_names,
                    y=top_values
                )
            ],

            'layout': {
                'title': 'Top 10 Most Frequent Categories',
                'yaxis': {
                    'title': "Frequency"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
