# import libraries
import sys
import re
import nltk
nltk.download(['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from xgboost import XGBClassifier
import pickle


def load_data(database_filepath):

    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterResponse', engine)
    #drop columns that are not going to use
    df = df.drop(['id', 'original', 'genre'], axis=1)
    X = np.asarray(df['message'])
    Y = np.asarray(df.drop(['message'], axis=1))
    category_names = [str(i) for i in df.columns[1:]]

    return X, Y, category_names


def tokenize(text):
    # remove punctuation and make all characters in lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize words
    tokens = word_tokenize(text)
    # remove stop words
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    lem = WordNetLemmatizer()
    # lemmatize every word
    clean_tokens = [lem.lemmatize(tok, pos='v') for tok in [lem.lemmatize(t) for t in tokens]]

    return clean_tokens


def build_model():

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(
            # use the xgboost classifier here
            XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            ))
        ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    for i in range(36):
        #evaluate each category
        print(f'category {category_names[i]}:\n{classification_report(Y_test[:,i], y_pred[:,i])}\n')


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()