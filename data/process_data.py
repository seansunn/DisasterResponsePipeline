import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    load_data
    load data from csv files and merge to a single dataframe on 'id'
    
    Input:
    messages_filepath       filepath to message.csv file
    categories_filepath     filepath to categories.csv file
    
    Returns:
    df      dataframe merging messages and categories
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge two dataframe together on 'id'
    df = pd.merge(categories, messages, on='id')
    
    return df


def clean_data(df):
    '''
    clean_data
    clean and prepare the merged dataframe for modeling
    
    Input:
    df      the merged dataframe from load_data function
    
    Returns:
    df      the cleaned dataframe
    '''
    # get category names from the first cell of the "categories" column
    lst = df['categories'][0].split(';')
    category_colnames = [i.split("-")[0] for i in lst]
    
    # split "categories" column into 36 columns, and drop column 'categories'
    df[category_colnames] = df['categories'].str.split(';', expand=True)
    df.drop(['categories'], axis=1, inplace=True)
    
    # clean values for new columns
    for column in category_colnames:
        
        # set each value to be the last character of the string; change type to integer
        df[column] = df[column].str.split("-").str.get(1).astype('int')
    
    # drop the rows in the 'related' column that are not 1 or 0
    df.drop(df[(df['related']!=1)&(df['related']!=0)].index, inplace=True)
    
    # drop duplicates
    df.drop_duplicates(subset='message', keep='first', inplace=True)
    
    return df


def save_data(df, database_filepath):
    '''
    save_data
    save the cleaned dataframe to sql database
    
    Input:
    df      the cleaned dataframe from clean_data function
    database_filepath       filepath for saving the database
    
    Returns:
    none
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql(database_filepath[:-3], engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:
    
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
