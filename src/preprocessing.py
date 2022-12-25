from re import X
from tkinter import Y
import pandas as pd
import numpy as np
import cyrtranslit
from sklearn.preprocessing import OneHotEncoder
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import datetime


def remove_unwanted_coulmns(data_frame:pd.DataFrame,
                            columns:str = ['Rbr', 'Dan', 'Datum', 'MOTORNI SAT']):
    """
    Remove unwanted columns from the dataframe.

    Args:
        data_frame: pd.DataFrame - the dataframe to be cleaned of unwanted columns
        columns: str - the columns to be removed from the dataframe

    Returns:
        pd.DataFrame - the cleaned dataframe    

    """
    data_frame.drop(columns, axis = 1, inplace = True)

    return data_frame

def remove_target_column(data_frame:pd.DataFrame,
                         target_column:str = 'Nor stepen opasnosti'):
    """
    Remove the target column from the dataframe and save it to a separate dataframe.

    Args:
        data_frame: pd.DataFrame   - Our dataframe
        target_column: str - the target column to be removed and saved seperately
    
    Returns:
        data_frame: pd.DataFrame - dataframe without the target column
        y: pd.DataFrame - the target column we are saving for later use
    """
    y = data_frame[target_column] # Save the target column for later use

    data_frame.drop(target_column, axis = 1, inplace = True)


    return data_frame, y

def  replace_nan_with_zeros_and_ones(data_frame: pd.DataFrame,
                                    columns:str = ['Tehnoloski', 'Elektro/struja', 'Mehanicki', 
                                                    'Zloupotreba', 'Organizacioni', 'Eksterni uticaj']):
    """
    Replace nan with zeros and ones in columns we want to turn to categorical.

    Args:
        data_frame: pd.DataFrame  
        columns: str - the columns to be turned to categorical

    Returns:
        pd.DataFrame - the cleaned dataframe    

    """
    data_frame[columns] = data_frame[columns].fillna(0)

    for column in columns:
        data_frame.loc[data_frame[column] != 0, column] = 1

    return data_frame

def fill_missing_values_based_on_similar_column(data_frame:pd.DataFrame,
                        column_w_nan:str = ['Masina', 'Uzrok'],
                        columns_to_fill:str = ['VRSTA MASINE', 'RAD']):
    """
    Fill missing values in column with similar values from another column.

    Args:
        data_frame: pd.DataFrame  
        column_w_nan: str - the columns that has missing values
        columns_to_fill: str - the column that will be used to fill the missing values in the column_w_nan
    
    Returns:
        pd.DataFrame - the dataframe with filled missing values in the column_w_nan  

    """
    counter = 0
    for counter in range(len(column_w_nan)):
        data_frame[column_w_nan[counter]] = data_frame[column_w_nan[counter]].fillna(data_frame[columns_to_fill[counter]])
        counter += 1

    return data_frame

def cyrilic_to_latin(data_frame:pd.DataFrame,
                     columns:str = ['Uzrok', 'RAD', 'VRSTA MASINE', 'Masina']):
    """
    Convert cyrilic text to latin text in selected columns.

    Args:
        data_frame: pd.DataFrame  
        columns: str - the columns to be converted to latin

    Returns:
        data_frame: pd.DataFrame - dataframe with all latin text
    """
    for column in columns:
        data_frame[column] = data_frame[column].apply(lambda x: cyrtranslit.to_latin(x) if type(x) == str else x)

    return data_frame

def lower_and_strip_all_data(data_frame:pd.DataFrame,
                             columns:str = ['Uzrok', 'RAD', 'VRSTA MASINE', 'Masina'],
                             target_column:str = 'Nor stepen opasnosti'):
    """
    Lower and strip all data in selected columns.

    Args:
        data_frame: pd.DataFrame  
        columns: str - the columns to be lower and stripped
    
    Returns:
        data_frame: pd.DataFrame - dataframe with all lower and stripped text
    """
    for column in columns:
        data_frame[column] = data_frame[column].apply(lambda x: x.lower().strip() if type(x) == str else x)

    return data_frame

def one_hot_encode_categorical_data(data_frame:pd.DataFrame,
                                        categorical:str = ['Masina', 'RAD', 'VRSTA MASINE', 'Uzrok', 'Tehnoloski','Elektro/struja', 'Mehanicki',
                                                            'Zloupotreba', 'Organizacioni', 'Eksterni uticaj'],
                                        pickle_file_name:str = 'one_hot_encoder',
                                        pickle_the_encoder:bool = False):
    """
    One hot encode selected columns and pickle the encoding for later use in the
    streamlit app.

    Args:
        data_frame: pd.DataFrame  
        categorical: str - the columns to be one hot encoded
        pickle_file_name: str - the name of the pickle file to save the one hot encoder
        pickle_the_encoder: bool - if True, pickle the one hot encoder

    Returns:
        OH_columns: pd.DataFrame - the one hot encoded dataframe
    """
    # Apply one-hot encoder to the relevant columns
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    OH_cols = pd.DataFrame(OH_encoder.fit_transform(data_frame[categorical]))
    
    # Replace default column names with more descriptive ones
    OH_cols.columns = OH_encoder.get_feature_names(categorical)

    OH_cols.index = data_frame.index
    OH_cols = OH_cols.reset_index(drop=True)

    if pickle_the_encoder==True:
        now = datetime.datetime.now()
        pickle_encder = open('{}_{}_{}.pkl'.format(pickle_file_name, now.month, now.day), 'wb')
        pickle.dump(OH_encoder, pickle_encder)
        pickle_encder.close()

    return OH_cols

def scale_numerical_data(data_frame:pd.DataFrame,
                         numerical:str = ['Vreme'],
                         scaler_file_name:str = 'standard_scaler',
                         pickle_the_scaler:bool = False):
    """
    Scale numerical data.

    Args:
        data_frame: pd.DataFrame  
        numerical: str - the columns to be scaled
        scaler_file_name: str - the name of the pickle file to save the standard scaler
        pickle_the_scaler: bool - if True, pickle the scaler for later use in the streamlit app
    
    Returns:
        num_scaled_dataset: pd.DataFrame - dataframe with scaled numerical columns.
    """
    numerical_values = data_frame[numerical].values
    scaler = StandardScaler()

    num_scaled = scaler.fit_transform(numerical_values.reshape(-1, 1))
    num_scaled = pd.DataFrame(num_scaled)

    num_scaled = num_scaled.reset_index(drop=True)

    if pickle_the_scaler==True:
        now = datetime.datetime.now()
        pickle_scaler = open('{}_{}_{}.pkl'.format(scaler_file_name, now.month, now.day), 'wb')
        pickle.dump(scaler, pickle_scaler)
        pickle_scaler.close()

    return num_scaled

def oh_and_scaled_data(oh_encoded_data:pd.DataFrame,
                       scaled_data:pd.DataFrame,
                       name_of_numerical_column:str = 'Vreme'):
    """
    Combine the one hot encoded dataframe with the scaled numerical dataframe.

    Args:
        oh_encoded_data: pd.DataFrame  - the one hot encoded dataframe
        scaled_data: pd.DataFrame - the scaled numerical dataframe
    
    Returns:
        combined_data: pd.DataFrame - dataframe with combined columns
    """
    combined_data = oh_encoded_data
    combined_data[name_of_numerical_column] = scaled_data

    return combined_data

def train_test_split_data(data_frame:pd.DataFrame,
                          target_data:pd.DataFrame,
                          test_size:float = 0.2,
                          random_state:int = 42):
    """
    Split the data into train and test sets.

    Args:
        data_frame: pd.DataFrame  
        test_size: float - the size of the test set
        random_state: int - the random state to be used for splitting the data
    
    Returns:
        train_data: pd.DataFrame - dataframe with train data
        test_data: pd.DataFrame - dataframe with test data
    """
    X = data_frame
    y = target_data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

def preprocess_data_for_training_and_testing(data_frame:pd.DataFrame):
    """
    Preprocess data for training and testing.

    Args:
        data_frame: pd.DataFrame

    Returns:
        X_train, X_test, y_train, y_test: pd.DataFrame - dataframes with preprocessed data ready for training and testing
    """
    data = remove_unwanted_coulmns(data_frame)
    data, y = remove_target_column(data)
    data = replace_nan_with_zeros_and_ones(data)
    data = fill_missing_values_based_on_similar_column(data)
    data = cyrilic_to_latin(data)
    data = lower_and_strip_all_data(data)
    OH_cols = one_hot_encode_categorical_data(data, pickle_the_encoder=False)
    num_scaled = scale_numerical_data(data, pickle_the_scaler=False)
    data = oh_and_scaled_data(OH_cols, num_scaled)
    X_train, X_test, y_train, y_test = train_test_split_data(data, y)

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    data = pd.read_excel("D:\Programing projects\Master_rad\data\Podaci_za_sve_masine.xlsx")
    X_train, X_test, y_train, y_test = preprocess_data_for_training_and_testing(data)