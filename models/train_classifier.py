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
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier
import pickle


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


def load_data(database_filepath):
    '''
    load_data
    load the cleaned data from the sql database into dataframe
    
    Input:
    database_filepath       the filepath of sql database
    
    Returns:
    X   independent variable; text messages
    Y   dependent variable; categories
    category_names      list containing the names of the 36 categories
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterResponse', engine)
    
    # drop columns that are not going to use
    df = df.drop(['id', 'original', 'genre'], axis=1)
    
    # independent variable
    X = np.asarray(df['message'])
    
    # dependent variable
    Y = np.asarray(df.drop(['message'], axis=1))
    
    # extract category names
    category_names = [str(i) for i in df.columns[1:]]

    return X, Y, category_names


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


def build_model():
    '''
    build_model
    build the pipeline for training the model
    feature extraction from texts and then classification using XGBoost
    
    Input:
    none
    
    Returns:
    pipeline    pipeline for training the model
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('txt_pipeline', Pipeline([
                
                # CountVectorizer using customized function tokenizer
                ('vect', CountVectorizer(tokenizer=tokenize)),
                
                # TfidfTransformer from sklearn
                ('tfidf', TfidfTransformer())
            ])),
            
            # customized class StartingVerbExtractor
            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(
            
            # use the xgboost classifier here
            XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            ))
        ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate_model
    evaluate the performance of the model through f1 score, 
    percision, and accuracy on each category
    
    Input:
    model       trained model
    X_test      test set for X
    Y_test      test set for Y
    category_names  list of all category names
    
    Returns:
    none
    '''
    # predict using the test set
    y_pred = model.predict(X_test)
    
    for i in range(36):
        
        #evaluate each category
        print(f'category {category_names[i]}:\n{classification_report(Y_test[:,i], y_pred[:,i])}\n')


def save_model(model, model_filepath):
    '''
    save_model
    use pickle to save the trained model
    
    Input:
    model       trained model
    model_filepath      path to save the model
    
    Returns:
    none
    '''
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
